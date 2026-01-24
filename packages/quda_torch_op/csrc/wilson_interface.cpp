/**
 * QUDA Wilson Dslash solver interface for quda_torch_op
 *
 * This file implements the interface between PyTorch tensors and QUDA's
 * Wilson fermion solver (invertQuda).
 *
 * REQUIREMENTS:
 * =============
 * QUDA must be built with at least one gauge field interface enabled.
 * The default CMake options enable QDP and MILC interfaces (recommended):
 *   cmake -DQUDA_INTERFACE_QDP=ON -DQUDA_INTERFACE_MILC=ON ...
 *
 * This implementation uses QUDA_QDP_GAUGE_ORDER which is enabled by default.
 *
 * Storage Format Conversions:
 * ==========================
 *
 * Gauge Field:
 * - plaq format: [4, V, 3, 3] where V = Nx*Ny*Nz*Nt (lexicographic site order)
 *   - First index is direction mu (0=x, 1=y, 2=z, 3=t)
 *   - Second index is site in lexicographic order (x varies fastest)
 *   - Last two indices are color matrix (row, col)
 *
 * - QUDA QDP format: gauge[mu] points to [V, 3, 3] for each mu
 *   - Sites in even-odd ordering (even sites first, then odd sites)
 *   - parity(x,y,z,t) = (x+y+z+t) % 2; even=0, odd=1
 *   - Color matrix in row-column order (same as plaq)
 *
 * Spinor Field:
 * - plaq format: [V, 4, 3] (site, spin, color)
 * - QUDA format: [V, 4, 3] with even-odd site ordering
 *   - For full solve: QUDA uses same layout but with EO reordering
 *
 * Gamma Matrix Convention:
 * - plaq uses MILC convention which maps to QUDA_DEGRAND_ROSSI_GAMMA_BASIS
 *
 * Data Copy Notes:
 * ================
 * - Gauge field: Requires site reordering from lexicographic to even-odd.
 *   Copy is necessary because plaq uses lexicographic order while QUDA
 *   requires even-odd ordering for optimal GPU performance.
 * - Spinor field: Also requires site reordering (same reason).
 * - Future optimization: If plaq stored data in even-odd order natively,
 *   we could pass pointers directly to QUDA (zero-copy).
 */

#include <Python.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <quda.h>
#include <array>
#include <complex>
#include <cstring>
#include <vector>

namespace quda_torch_op {

// Forward declaration from quda_interface.cpp
extern bool quda_is_initialized();

/**
 * Reorder sites from lexicographic to even-odd ordering.
 *
 * Given lattice dimensions [Nx, Ny, Nz, Nt], the lexicographic site index is:
 *   site = x + Nx*(y + Ny*(z + Nz*t))
 *
 * The even-odd ordering puts all even-parity sites first, then odd-parity sites.
 * Parity is defined as: parity = (x + y + z + t) % 2
 *
 * @param dims Lattice dimensions [Nx, Ny, Nz, Nt]
 * @return Vector of length V mapping even-odd index to lexicographic index
 */
std::vector<int64_t> compute_eo_to_lex_map(const int* dims) {
    int64_t V = dims[0] * dims[1] * dims[2] * dims[3];
    int64_t V_half = V / 2;
    std::vector<int64_t> eo_to_lex(V);

    int64_t even_idx = 0;
    int64_t odd_idx = V_half;

    for (int t = 0; t < dims[3]; t++) {
        for (int z = 0; z < dims[2]; z++) {
            for (int y = 0; y < dims[1]; y++) {
                for (int x = 0; x < dims[0]; x++) {
                    int64_t lex_site = x + dims[0] * (y + dims[1] * (z + dims[2] * t));
                    int parity = (x + y + z + t) % 2;
                    if (parity == 0) {
                        eo_to_lex[even_idx++] = lex_site;
                    } else {
                        eo_to_lex[odd_idx++] = lex_site;
                    }
                }
            }
        }
    }

    return eo_to_lex;
}

/**
 * Reorder sites from even-odd ordering back to lexicographic ordering.
 *
 * @param dims Lattice dimensions [Nx, Ny, Nz, Nt]
 * @return Vector of length V mapping lexicographic index to even-odd index
 */
std::vector<int64_t> compute_lex_to_eo_map(const int* dims) {
    auto eo_to_lex = compute_eo_to_lex_map(dims);
    int64_t V = dims[0] * dims[1] * dims[2] * dims[3];
    std::vector<int64_t> lex_to_eo(V);

    for (int64_t eo_idx = 0; eo_idx < V; eo_idx++) {
        lex_to_eo[eo_to_lex[eo_idx]] = eo_idx;
    }

    return lex_to_eo;
}

/**
 * Structure to hold QDP gauge field pointers.
 * QDP format uses an array of 4 pointers, one for each direction.
 */
struct QDPGaugeField {
    void* data;       // Single contiguous allocation
    void* ptrs[4];    // Pointers to each direction's data
};

/**
 * Convert gauge field from plaq format to QUDA QDP format.
 *
 * plaq: [4, V, 3, 3] complex, lexicographic site order (mu, site, row, col)
 * QDP: gauge[mu] -> [V, 3, 3] complex, even-odd site order (same row-col as plaq)
 *
 * The conversion requires site reordering from lexicographic to even-odd because
 * QUDA's GPU kernels are optimized for even-odd ordering.
 *
 * T should be the underlying real type (float or double).
 * Complex numbers are stored as pairs of T (real, imag).
 *
 * @param gauge_plaq Input gauge field tensor [4, V, 3, 3]
 * @param dims Lattice dimensions [Nx, Ny, Nz, Nt]
 * @return QDPGaugeField with data and pointer array
 */
template<typename T>
QDPGaugeField convert_gauge_plaq_to_quda_qdp(
    const torch::stable::Tensor& gauge_plaq,
    const int* dims
) {
    int64_t V = dims[0] * dims[1] * dims[2] * dims[3];
    auto lex_to_eo = compute_lex_to_eo_map(dims);

    // QDP format: 4 separate [V, 3, 3] arrays, one per direction
    // Total: 4 * V * 9 complex = 4*V*18 T values
    size_t bytes_per_direction = V * 9 * 2 * sizeof(T);  // V * 9 complex
    size_t total_size = 4 * bytes_per_direction;

    QDPGaugeField result;
    result.data = malloc(total_size);
    T* dst = static_cast<T*>(result.data);

    // Set up pointers for each direction
    for (int mu = 0; mu < 4; mu++) {
        result.ptrs[mu] = dst + mu * V * 18;
    }

    // Access raw bytes, then cast to T*
    const T* src = reinterpret_cast<const T*>(
        static_cast<const char*>(gauge_plaq.const_data_ptr())
    );

    // plaq format: [4, V, 3, 3] = [mu, lex_site, row, col]
    // QDP format: gauge[mu][eo_site, row, col] - same row-col order as plaq
    // Only need to reorder sites from lexicographic to even-odd

    for (int mu = 0; mu < 4; mu++) {
        T* dst_mu = static_cast<T*>(result.ptrs[mu]);
        const T* src_mu = src + mu * V * 18;  // 18 = 9 complex = 9*2 real

        for (int64_t lex_site = 0; lex_site < V; lex_site++) {
            int64_t eo_site = lex_to_eo[lex_site];
            // Copy entire color matrix (9 complex = 18 T values) per site
            memcpy(dst_mu + eo_site * 18, src_mu + lex_site * 18, 18 * sizeof(T));
        }
    }

    return result;
}

/**
 * Convert spinor field from plaq format to QUDA format.
 *
 * plaq: [V, 4, 3] complex, lexicographic site order
 * QUDA: [V, 4, 3] complex, even-odd site order (space-spin-color)
 *
 * T should be the underlying real type (float or double).
 *
 * @param spinor_plaq Input spinor tensor [V, 4, 3]
 * @param dims Lattice dimensions [Nx, Ny, Nz, Nt]
 * @return Pointer to converted spinor data (caller owns memory)
 */
template<typename T>
void* convert_spinor_plaq_to_quda(
    const torch::stable::Tensor& spinor_plaq,
    const int* dims
) {
    int64_t V = dims[0] * dims[1] * dims[2] * dims[3];
    auto lex_to_eo = compute_lex_to_eo_map(dims);

    // V * 4 * 3 complex = V * 12 complex = V * 24 T values
    size_t values_per_site = 24;  // 12 complex = 24 T values
    size_t bytes = V * values_per_site * sizeof(T);
    void* spinor_quda = malloc(bytes);
    T* dst = static_cast<T*>(spinor_quda);

    // Access raw bytes, then cast to T*
    const T* src = reinterpret_cast<const T*>(
        static_cast<const char*>(spinor_plaq.const_data_ptr())
    );

    // QUDA expects spinor in space-spin-color order with EO ordering
    // plaq stores [V, 4, 3] which is already space-spin-color
    for (int64_t lex_site = 0; lex_site < V; lex_site++) {
        int64_t eo_site = lex_to_eo[lex_site];
        // Copy 4*3 = 12 complex = 24 T values per site
        for (int i = 0; i < 24; i++) {
            dst[eo_site * 24 + i] = src[lex_site * 24 + i];
        }
    }

    return spinor_quda;
}

/**
 * Convert spinor field from QUDA format back to plaq format.
 *
 * T should be the underlying real type (float or double).
 *
 * @param spinor_quda QUDA spinor data pointer
 * @param dims Lattice dimensions [Nx, Ny, Nz, Nt]
 * @param source_tensor Source tensor (used to get dtype/device for output)
 * @return Tensor in plaq format [V, 4, 3]
 */
template<typename T>
torch::stable::Tensor convert_spinor_quda_to_plaq(
    const void* spinor_quda,
    const int* dims,
    const torch::stable::Tensor& source_tensor
) {
    int64_t V = dims[0] * dims[1] * dims[2] * dims[3];
    auto eo_to_lex = compute_eo_to_lex_map(dims);

    // Create output tensor [V, 4, 3] with same dtype/device as source
    std::array<int64_t, 3> sizes = {V, 4, 3};
    torch::stable::Tensor spinor_plaq = torch::stable::new_empty(
        source_tensor,
        torch::headeronly::IntHeaderOnlyArrayRef(sizes.data(), sizes.size())
    );

    // Access raw bytes, then cast to T*
    T* dst = reinterpret_cast<T*>(
        static_cast<char*>(spinor_plaq.mutable_data_ptr())
    );
    const T* src = static_cast<const T*>(spinor_quda);

    // Convert from even-odd back to lexicographic
    for (int64_t eo_site = 0; eo_site < V; eo_site++) {
        int64_t lex_site = eo_to_lex[eo_site];
        // Copy 12 complex = 24 T values per site
        for (int i = 0; i < 24; i++) {
            dst[lex_site * 24 + i] = src[eo_site * 24 + i];
        }
    }

    return spinor_plaq;
}

/**
 * Free gauge field memory allocated by convert_gauge_plaq_to_quda_qdp.
 */
void free_gauge_qdp(QDPGaugeField& gauge) {
    if (gauge.data) {
        free(gauge.data);
        gauge.data = nullptr;
        for (int i = 0; i < 4; i++) {
            gauge.ptrs[i] = nullptr;
        }
    }
}

/**
 * Solve the Wilson Dirac equation using QUDA.
 *
 * Solves either:
 *   - M*x = b              (equation="M")
 *   - M^dag*M*x = M^dag*b  (equation="MdagM")
 *
 * @param gauge Gauge field tensor [4, V, 3, 3] in plaq format
 * @param source Source spinor tensor [V, 4, 3] in plaq format
 * @param dims Lattice dimensions tensor [4] containing [Nx, Ny, Nz, Nt]
 * @param kappa Hopping parameter kappa = 1/(2*(m0 + 4r))
 * @param tol Solver tolerance
 * @param maxiter Maximum number of iterations
 * @param equation Equation type: "M" or "MdagM"
 * @param t_boundary Temporal boundary condition: -1 for antiperiodic, +1 for periodic
 * @return Tuple of (solution tensor [V, 4, 3], converged bool, iterations int, residual double)
 */
std::tuple<torch::stable::Tensor, bool, int64_t, double> wilson_invert(
    const torch::stable::Tensor& gauge,
    const torch::stable::Tensor& source,
    const torch::stable::Tensor& dims_tensor,
    double kappa,
    double tol,
    int64_t maxiter,
    const std::string& equation,
    int64_t t_boundary
) {
    // Validate QUDA is initialized
    STD_TORCH_CHECK(quda_is_initialized(),
        "QUDA is not initialized. Call quda_init() first.");

    // Validate tensor properties
    STD_TORCH_CHECK(gauge.device().type() == torch::headeronly::DeviceType::CPU,
        "Gauge field must be on CPU");
    STD_TORCH_CHECK(source.device().type() == torch::headeronly::DeviceType::CPU,
        "Source spinor must be on CPU");

    auto gauge_dtype = gauge.scalar_type();
    auto source_dtype = source.scalar_type();
    STD_TORCH_CHECK(gauge_dtype == source_dtype,
        "Gauge and source must have same dtype");

    bool is_double = (gauge_dtype == torch::headeronly::ScalarType::ComplexDouble);
    STD_TORCH_CHECK(is_double || gauge_dtype == torch::headeronly::ScalarType::ComplexFloat,
        "Only complex64 and complex128 dtypes are supported");

    // Extract lattice dimensions
    auto dims_contig = torch::stable::contiguous(dims_tensor);
    const int64_t* dims_ptr = dims_contig.const_data_ptr<int64_t>();
    int dims[4] = {
        static_cast<int>(dims_ptr[0]),
        static_cast<int>(dims_ptr[1]),
        static_cast<int>(dims_ptr[2]),
        static_cast<int>(dims_ptr[3])
    };
    int64_t V = dims[0] * dims[1] * dims[2] * dims[3];

    // Validate tensor shapes
    STD_TORCH_CHECK(gauge.sizes().equals({4, V, 3, 3}),
        "Gauge field must have shape [4, V, 3, 3]");
    STD_TORCH_CHECK(source.sizes().equals({V, 4, 3}),
        "Source spinor must have shape [V, 4, 3]");

    // Make tensors contiguous
    auto gauge_contig = torch::stable::contiguous(gauge);
    auto source_contig = torch::stable::contiguous(source);

    // Set up QUDA gauge parameters
    QudaGaugeParam gauge_param = newQudaGaugeParam();
    gauge_param.struct_size = sizeof(gauge_param);

    gauge_param.X[0] = dims[0];
    gauge_param.X[1] = dims[1];
    gauge_param.X[2] = dims[2];
    gauge_param.X[3] = dims[3];

    gauge_param.type = QUDA_WILSON_LINKS;
    gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
    gauge_param.t_boundary = (t_boundary == -1) ? QUDA_ANTI_PERIODIC_T : QUDA_PERIODIC_T;

    gauge_param.cpu_prec = is_double ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
    gauge_param.cuda_prec = is_double ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
    gauge_param.cuda_prec_sloppy = is_double ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
    gauge_param.cuda_prec_precondition = is_double ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;

    gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
    gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    gauge_param.reconstruct_precondition = QUDA_RECONSTRUCT_NO;

    gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;
    gauge_param.anisotropy = 1.0;
    gauge_param.ga_pad = 0;

    gauge_param.location = QUDA_CPU_FIELD_LOCATION;

    // Set up QUDA invert parameters
    QudaInvertParam inv_param = newQudaInvertParam();
    inv_param.struct_size = sizeof(inv_param);

    inv_param.dslash_type = QUDA_WILSON_DSLASH;
    inv_param.kappa = kappa;

    inv_param.cpu_prec = is_double ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
    inv_param.cuda_prec = is_double ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
    inv_param.cuda_prec_sloppy = is_double ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
    inv_param.cuda_prec_precondition = is_double ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;

    // Dirac field order: QDP uses spin inside color
    inv_param.dirac_order = QUDA_DIRAC_ORDER;

    // MILC gamma matrices correspond to DeGrand-Rossi basis in QUDA
    inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

    inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
    inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

    // Solver configuration
    if (equation == "M") {
        // Direct solve: M*x = b using BiCGStab
        inv_param.inv_type = QUDA_BICGSTAB_INVERTER;
        inv_param.solution_type = QUDA_MAT_SOLUTION;
        inv_param.solve_type = QUDA_DIRECT_SOLVE;
        inv_param.dagger = QUDA_DAG_NO;
    } else if (equation == "MdagM") {
        // Normal equation: M^dag*M*x = M^dag*b using CG
        inv_param.inv_type = QUDA_CG_INVERTER;
        inv_param.solution_type = QUDA_MATDAG_MAT_SOLUTION;
        inv_param.solve_type = QUDA_NORMOP_SOLVE;
        inv_param.dagger = QUDA_DAG_NO;
    } else {
        STD_TORCH_CHECK(false, "equation must be 'M' or 'MdagM'");
    }

    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
    inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;

    inv_param.tol = tol;
    inv_param.maxiter = static_cast<int>(maxiter);
    inv_param.reliable_delta = 1e-4;

    inv_param.use_init_guess = QUDA_USE_INIT_GUESS_NO;
    inv_param.compute_true_res = 1;
    inv_param.residual_type = QUDA_L2_RELATIVE_RESIDUAL;

    inv_param.verbosity = QUDA_SILENT;

    // Allocate and convert data to QUDA format
    QDPGaugeField gauge_qdp = {};
    void* source_quda = nullptr;
    void* solution_quda = nullptr;

    try {
        if (is_double) {
            gauge_qdp = convert_gauge_plaq_to_quda_qdp<double>(gauge_contig, dims);
            source_quda = convert_spinor_plaq_to_quda<double>(source_contig, dims);
        } else {
            gauge_qdp = convert_gauge_plaq_to_quda_qdp<float>(gauge_contig, dims);
            source_quda = convert_spinor_plaq_to_quda<float>(source_contig, dims);
        }

        // Allocate solution buffer (same size as source)
        size_t spinor_bytes = V * 24 * (is_double ? sizeof(double) : sizeof(float));
        solution_quda = malloc(spinor_bytes);
        memset(solution_quda, 0, spinor_bytes);

        // Load gauge field into QUDA (QDP format uses array of pointers)
        loadGaugeQuda(gauge_qdp.ptrs, &gauge_param);

        // Perform the solve
        invertQuda(solution_quda, source_quda, &inv_param);

        // Free gauge field from QUDA
        freeGaugeQuda();

        // Convert solution back to plaq format
        torch::stable::Tensor solution;
        if (is_double) {
            solution = convert_spinor_quda_to_plaq<double>(
                solution_quda, dims, source_contig);
        } else {
            solution = convert_spinor_quda_to_plaq<float>(
                solution_quda, dims, source_contig);
        }

        // Extract solver statistics
        bool converged = (inv_param.true_res[0] < tol);
        int64_t iterations = inv_param.iter;
        double residual = inv_param.true_res[0];

        // Clean up
        free_gauge_qdp(gauge_qdp);
        if (source_quda) free(source_quda);
        if (solution_quda) free(solution_quda);

        return std::make_tuple(solution, converged, iterations, residual);

    } catch (...) {
        // Clean up on error
        free_gauge_qdp(gauge_qdp);
        if (source_quda) free(source_quda);
        if (solution_quda) free(solution_quda);
        throw;
    }
}

// Define operator schema in the STABLE_TORCH_LIBRARY block in code.cpp
// Register implementation here

STABLE_TORCH_LIBRARY_IMPL(quda_torch_op, CompositeExplicitAutograd, m) {
    m.impl("wilson_invert", TORCH_BOX(&wilson_invert));
}

}  // namespace quda_torch_op
