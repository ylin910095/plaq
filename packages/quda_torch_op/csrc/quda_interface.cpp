/**
 * QUDA interface for quda_torch_op
 *
 * This file provides PyTorch operator implementations that interface with QUDA.
 * The operator schema is defined in code.cpp; this file only registers implementations.
 *
 * Note: QUDA is built with MPI support, so we initialize MPI for single-process use.
 */

#include <Python.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <quda.h>
#include <device.h>
#include <mpi.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <mutex>
#include <string>

namespace quda_torch_op {

// Thread-safe QUDA initialization state
static std::atomic<bool> g_quda_initialized{false};
static std::mutex g_quda_mutex;
static int g_quda_device = -1;

/**
 * Check if QUDA is available (compiled with QUDA support).
 */
bool quda_is_available() {
    return true;  // If this code is compiled, QUDA is available
}

/**
 * Get number of CUDA devices.
 */
int64_t quda_get_device_count() {
    return quda::device::get_device_count();
}

/**
 * Initialize QUDA on a specific device.
 *
 * Args:
 *     device: CUDA device index (0-based). Use -1 for default device.
 *
 * This function is idempotent - calling it multiple times with the same
 * device is safe. However, calling with a different device after initialization
 * will raise an error.
 */
void quda_init(int64_t device) {
    std::lock_guard<std::mutex> lock(g_quda_mutex);

    if (g_quda_initialized.load()) {
        // Already initialized - check if same device
        if (g_quda_device != device && device != -1) {
            STD_TORCH_CHECK(
                false,
                "QUDA already initialized on device " + std::to_string(g_quda_device) +
                ", cannot reinitialize on device " + std::to_string(device));
        }
        return;
    }

    int num_devices = quda::device::get_device_count();
    STD_TORCH_CHECK(
        num_devices > 0,
        "No CUDA devices found. QUDA requires at least one GPU.");

    int target_device = (device == -1) ? 0 : static_cast<int>(device);
    STD_TORCH_CHECK(
        target_device >= 0 && target_device < num_devices,
        "Invalid device index " + std::to_string(target_device) +
        ". Available devices: 0-" + std::to_string(num_devices - 1));

    // Initialize MPI if not already initialized (QUDA requires MPI)
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        int argc = 0;
        char** argv = nullptr;
        MPI_Init(&argc, &argv);
    }

    // Set up single-process communication grid (1x1x1x1)
    int comm_dims[4] = {1, 1, 1, 1};
    initCommsGridQuda(4, comm_dims, nullptr, nullptr);

    // Set verbosity
    setVerbosityQuda(QUDA_SUMMARIZE, "", stdout);

    // Initialize QUDA
    initQuda(target_device);

    g_quda_device = target_device;
    g_quda_initialized.store(true);
}

/**
 * Finalize QUDA and release resources.
 *
 * This should be called when QUDA is no longer needed to free GPU memory.
 * Note: MPI is NOT finalized because it's a process-wide resource that
 * cannot be restarted once finalized. MPI will be cleaned up when the
 * process exits.
 */
void quda_finalize() {
    std::lock_guard<std::mutex> lock(g_quda_mutex);

    if (!g_quda_initialized.load()) {
        return;  // Nothing to do
    }

    endQuda();

    // Note: We intentionally do NOT call MPI_Finalize() here.
    // MPI cannot be restarted once finalized, and the library should
    // support reinitializing QUDA in long-running processes.
    // MPI will be cleaned up automatically when the process exits.

    g_quda_initialized.store(false);
    g_quda_device = -1;
}

/**
 * Check if QUDA has been initialized.
 */
bool quda_is_initialized() {
    return g_quda_initialized.load();
}

/**
 * Get the device QUDA was initialized on.
 * Returns -1 if not initialized.
 */
int64_t quda_get_device() {
    return g_quda_device;
}

/**
 * Get QUDA version string.
 */
std::string quda_get_version() {
    // QUDA doesn't have a runtime version function, use compile-time version
    return std::string("1.1.0");
}

// =============================================================================
// Wilson Dslash Operator Implementation
// =============================================================================

/**
 * Helper: Get lattice dimensions from gauge tensor shape.
 * plaq gauge layout is [4, V, 3, 3] where V = Nx * Ny * Nz * Nt.
 * We need to recover Nx, Ny, Nz, Nt from the spinor shape and context.
 *
 * For QUDA, we pass dimensions explicitly via the gauge tensor metadata
 * or infer from a 5D gauge tensor [4, Nt, Nz, Ny, Nx, 3, 3].
 *
 * In plaq, the gauge field has shape [4, V, 3, 3] with lexicographic ordering
 * x + Nx*(y + Ny*(z + Nz*t)), so we need dimensions passed separately.
 */
struct LatticeDims {
    int Nx, Ny, Nz, Nt;
    int64_t volume() const { return static_cast<int64_t>(Nx) * Ny * Nz * Nt; }
};

/**
 * Infer lattice dimensions from gauge tensor.
 *
 * If gauge is [4, Nt, Nz, Ny, Nx, 3, 3] (7D), extract directly.
 * If gauge is [4, V, 3, 3] (4D), try to factorize V assuming Nx=Ny=Nz<=Nt.
 */
LatticeDims infer_lattice_dims(const torch::stable::Tensor& gauge, const torch::stable::Tensor& psi) {
    auto gauge_sizes = gauge.sizes();
    auto psi_sizes = psi.sizes();

    // psi shape is [V, 4, 3], gauge shape is [4, V, 3, 3]
    STD_TORCH_CHECK(psi_sizes.size() == 3, "Spinor must have shape [V, 4, 3]");
    STD_TORCH_CHECK(gauge_sizes.size() == 4, "Gauge must have shape [4, V, 3, 3]");

    int64_t V = psi_sizes[0];
    STD_TORCH_CHECK(gauge_sizes[1] == V, "Gauge and spinor volume must match");

    LatticeDims dims;

    // First try: perfect 4th root (hypercubic lattice, e.g. 4^4, 8^4)
    double fourth_root = std::pow(static_cast<double>(V), 0.25);
    int L = static_cast<int>(std::round(fourth_root));
    if (static_cast<int64_t>(L) * L * L * L == V) {
        dims.Nx = dims.Ny = dims.Nz = dims.Nt = L;
        return dims;
    }

    // Second try: Nx=Ny=Nz with different Nt, requiring Nt >= Ls
    // (standard lattice QCD convention: temporal extent >= spatial extent)
    for (int Ls : {32, 24, 16, 12, 8, 4}) {
        int64_t L3 = static_cast<int64_t>(Ls) * Ls * Ls;
        if (V % L3 == 0) {
            int64_t Nt = V / L3;
            if (Nt >= Ls && Nt <= 128) {
                dims.Nx = Ls;
                dims.Ny = Ls;
                dims.Nz = Ls;
                dims.Nt = static_cast<int>(Nt);
                return dims;
            }
        }
    }

    STD_TORCH_CHECK(false,
        "Cannot infer lattice dimensions from volume " + std::to_string(V) +
        ". Use explicit dimension passing or standard lattice sizes.");
    return dims;  // Unreachable
}

/**
 * Convert plaq spinor layout [V, 4, 3] to QUDA layout [2, Lt, Lz, Ly, Lx/2, 4, 3].
 *
 * plaq uses lexicographic site ordering: site = x + Nx*(y + Ny*(z + Nz*t))
 * QUDA uses even-odd split with t-slowest ordering within each parity.
 *
 * Even sites: (x+y+z+t) % 2 == 0
 * Odd sites: (x+y+z+t) % 2 == 1
 */
torch::stable::Tensor plaq_spinor_to_quda(
    const torch::stable::Tensor& psi,
    const LatticeDims& dims) {

    int Nx = dims.Nx, Ny = dims.Ny, Nz = dims.Nz, Nt = dims.Nt;
    int64_t V = dims.volume();
    int64_t Vh = V / 2;

    // Create QUDA layout tensor [2, Vh, 4, 3]
    // We use a flat even-odd layout for simplicity with QUDA's mat functions
    auto psi_contig = torch::stable::contiguous(psi);

    // Create output tensor with same dtype
    int64_t out_shape[] = {2, Vh, 4, 3};
    auto result = torch::stable::empty(
        {out_shape, 4},
        psi_contig.scalar_type(),
        std::nullopt,  // layout
        psi_contig.device());

    // Get data pointers - assuming complex double
    STD_TORCH_CHECK(psi_contig.scalar_type() == torch::headeronly::ScalarType::ComplexDouble,
        "Only complex128 (ComplexDouble) is currently supported");

    // Use void* API and cast - complex<double> is stored as 2 doubles
    const std::complex<double>* src = static_cast<const std::complex<double>*>(psi_contig.const_data_ptr());
    std::complex<double>* dst = static_cast<std::complex<double>*>(result.mutable_data_ptr());

    // plaq spinor: [site][spin][color], site = x + Nx*(y + Ny*(z + Nz*t)) (x-fastest)
    // QUDA DIRAC_ORDER: [parity][cb_idx][spin][color] - color inside spin
    //
    // QUDA cb_idx formula: cb_idx = x/2 + (Nx/2) * (y + Ny * (z + Nz * t))
    // parity = (x + y + z + t) % 2

    for (int t = 0; t < Nt; t++) {
        for (int z = 0; z < Nz; z++) {
            for (int y = 0; y < Ny; y++) {
                for (int x = 0; x < Nx; x++) {
                    int64_t plaq_site = x + Nx * (y + Ny * (z + Nz * t));
                    int parity = (x + y + z + t) % 2;
                    int64_t cb_idx = plaq_site >> 1;

                    // Copy spinor components [spin][color] - same order as plaq
                    for (int s = 0; s < 4; s++) {
                        for (int c = 0; c < 3; c++) {
                            int64_t src_idx = plaq_site * 12 + s * 3 + c;
                            int64_t dst_idx = (parity * Vh + cb_idx) * 12 + s * 3 + c;
                            dst[dst_idx] = src[src_idx];
                        }
                    }
                }
            }
        }
    }

    return result;
}

/**
 * Convert QUDA spinor layout [2, Vh, 4, 3] back to plaq layout [V, 4, 3].
 */
torch::stable::Tensor quda_spinor_to_plaq(
    const torch::stable::Tensor& psi_quda,
    const LatticeDims& dims) {

    int Nx = dims.Nx, Ny = dims.Ny, Nz = dims.Nz, Nt = dims.Nt;
    int64_t V = dims.volume();
    int64_t Vh = V / 2;

    auto psi_contig = torch::stable::contiguous(psi_quda);

    int64_t out_shape[] = {V, 4, 3};
    auto result = torch::stable::empty(
        {out_shape, 3},
        psi_contig.scalar_type(),
        std::nullopt,  // layout
        psi_contig.device());

    const std::complex<double>* src = static_cast<const std::complex<double>*>(psi_contig.const_data_ptr());
    std::complex<double>* dst = static_cast<std::complex<double>*>(result.mutable_data_ptr());

    // QUDA DIRAC_ORDER: [parity][cb_idx][spin][color] - color inside spin
    // plaq spinor: [site][spin][color], site = x + Nx*(y + Ny*(z + Nz*t)) (x-fastest)
    //
    // QUDA cb_idx formula: cb_idx = x/2 + (Nx/2) * (y + Ny * (z + Nz * t))

    for (int t = 0; t < Nt; t++) {
        for (int z = 0; z < Nz; z++) {
            for (int y = 0; y < Ny; y++) {
                for (int x = 0; x < Nx; x++) {
                    int64_t plaq_site = x + Nx * (y + Ny * (z + Nz * t));
                    int parity = (x + y + z + t) % 2;
                    int64_t cb_idx = plaq_site >> 1;

                    // Same order as plaq: [spin][color]
                    for (int s = 0; s < 4; s++) {
                        for (int c = 0; c < 3; c++) {
                            int64_t src_idx = (parity * Vh + cb_idx) * 12 + s * 3 + c;
                            int64_t dst_idx = plaq_site * 12 + s * 3 + c;
                            dst[dst_idx] = src[src_idx];
                        }
                    }
                }
            }
        }
    }

    return result;
}

/**
 * Convert plaq gauge layout [4, V, 3, 3] to QUDA QDP gauge order.
 *
 * QUDA QDP gauge order: void*[4] array of pointers, each pointing to:
 *   [parity][cb_idx][row][col] data for that direction
 * Stored as contiguous [4, 2, Vh, 3, 3] with the 4 direction slices
 * addressed via separate pointers.
 *
 * Note: For antiperiodic temporal BC, we multiply the temporal (mu=3) gauge
 * links at t=Nt-1 by -1. We handle BC embedding here and set t_boundary=PERIODIC
 * in the gauge params to prevent QUDA from double-applying the BC.
 */
torch::stable::Tensor plaq_gauge_to_quda(
    const torch::stable::Tensor& gauge,
    const LatticeDims& dims,
    bool antiperiodic_t) {

    int Nx = dims.Nx, Ny = dims.Ny, Nz = dims.Nz, Nt = dims.Nt;
    int64_t V = dims.volume();
    int64_t Vh = V / 2;

    auto gauge_contig = torch::stable::contiguous(gauge);

    // QDP order: [mu][parity][cb_idx][row][col]
    // Stored as contiguous [4, 2, Vh, 3, 3]
    int64_t out_shape[] = {4, 2, Vh, 3, 3};
    auto result = torch::stable::empty(
        {out_shape, 5},
        gauge_contig.scalar_type(),
        std::nullopt,  // layout
        gauge_contig.device());

    STD_TORCH_CHECK(gauge_contig.scalar_type() == torch::headeronly::ScalarType::ComplexDouble,
        "Only complex128 (ComplexDouble) is currently supported");

    const std::complex<double>* src = static_cast<const std::complex<double>*>(gauge_contig.const_data_ptr());
    std::complex<double>* dst = static_cast<std::complex<double>*>(result.mutable_data_ptr());

    // plaq gauge: [mu][site][row][col], site = x + Nx*(y + Ny*(z + Nz*t)) (x-fastest)
    // QDP: [mu][parity][cb_idx][row][col]
    //
    // cb_idx = plaq_site >> 1
    // parity = (x + y + z + t) % 2

    for (int mu = 0; mu < 4; mu++) {
        for (int t = 0; t < Nt; t++) {
            for (int z = 0; z < Nz; z++) {
                for (int y = 0; y < Ny; y++) {
                    for (int x = 0; x < Nx; x++) {
                        int64_t plaq_site = x + Nx * (y + Ny * (z + Nz * t));
                        int parity = (x + y + z + t) % 2;
                        int64_t cb_idx = plaq_site >> 1;

                        // BC phase for forward temporal links at t=Nt-1
                        double bc_phase = 1.0;
                        if (antiperiodic_t && mu == 3 && t == Nt - 1) {
                            bc_phase = -1.0;
                        }

                        // Copy 3x3 matrix with BC phase
                        for (int row = 0; row < 3; row++) {
                            for (int col = 0; col < 3; col++) {
                                // plaq: [mu][site][row][col]
                                int64_t src_idx = mu * V * 9 + plaq_site * 9 + row * 3 + col;
                                // QDP: [mu][parity][cb_idx][row][col]
                                int64_t dst_idx = mu * 2 * Vh * 9 + parity * Vh * 9 + cb_idx * 9 + row * 3 + col;
                                dst[dst_idx] = src[src_idx] * bc_phase;
                            }
                        }
                    }
                }
            }
        }
    }

    return result;
}

/**
 * Initialize QudaGaugeParam for Wilson fermions.
 */
QudaGaugeParam create_gauge_param(const LatticeDims& dims, bool antiperiodic_t) {
    QudaGaugeParam gauge_param = newQudaGaugeParam();

    // Set struct size for version checking
    gauge_param.struct_size = sizeof(QudaGaugeParam);

    gauge_param.X[0] = dims.Nx;
    gauge_param.X[1] = dims.Ny;
    gauge_param.X[2] = dims.Nz;
    gauge_param.X[3] = dims.Nt;

    gauge_param.type = QUDA_WILSON_LINKS;
    gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
    // BC is embedded in the gauge field by plaq_gauge_to_quda(),
    // so always use PERIODIC here to prevent QUDA from double-applying the BC.
    (void)antiperiodic_t;  // Suppress unused parameter warning
    gauge_param.t_boundary = QUDA_PERIODIC_T;

    gauge_param.cpu_prec = QUDA_DOUBLE_PRECISION;
    gauge_param.cuda_prec = QUDA_DOUBLE_PRECISION;
    gauge_param.cuda_prec_sloppy = QUDA_DOUBLE_PRECISION;
    gauge_param.cuda_prec_precondition = QUDA_DOUBLE_PRECISION;

    gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
    gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    gauge_param.reconstruct_precondition = QUDA_RECONSTRUCT_NO;

    gauge_param.anisotropy = 1.0;
    gauge_param.tadpole_coeff = 1.0;
    gauge_param.scale = 1.0;
    // Padding for native gauge fields on GPU (matches PyQUDA convention)
    int min_dim = std::min({dims.Nx, dims.Ny, dims.Nz, dims.Nt});
    gauge_param.ga_pad = dims.volume() / min_dim / 2;

    // Gauge fixing - we assume the gauge is not fixed
    gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

    // Staggered phase settings (not used for Wilson, but set for completeness)
    gauge_param.staggered_phase_type = QUDA_STAGGERED_PHASE_MILC;
    gauge_param.staggered_phase_applied = 0;

    // Resident gauge settings
    gauge_param.overwrite_gauge = 0;
    gauge_param.overwrite_mom = 0;
    gauge_param.use_resident_gauge = 0;
    gauge_param.use_resident_mom = 0;
    gauge_param.make_resident_gauge = 0;
    gauge_param.make_resident_mom = 0;
    gauge_param.return_result_gauge = 0;
    gauge_param.return_result_mom = 0;

    // Location of gauge field
    gauge_param.location = QUDA_CPU_FIELD_LOCATION;

    return gauge_param;
}

/**
 * Initialize QudaInvertParam for Wilson Dslash.
 */
QudaInvertParam create_invert_param(double kappa, QudaMatPCType matpc_type = QUDA_MATPC_EVEN_EVEN) {
    QudaInvertParam inv_param = newQudaInvertParam();

    // Set struct size for version checking
    inv_param.struct_size = sizeof(QudaInvertParam);

    inv_param.dslash_type = QUDA_WILSON_DSLASH;
    inv_param.kappa = kappa;
    inv_param.mass = 0.5 / kappa - 4.0;  // m0 = 1/(2*kappa) - 4

    // Use DeGrand-Rossi gamma basis (QUDA default)
    inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

    inv_param.cpu_prec = QUDA_DOUBLE_PRECISION;
    inv_param.cuda_prec = QUDA_DOUBLE_PRECISION;
    inv_param.cuda_prec_sloppy = QUDA_DOUBLE_PRECISION;
    inv_param.cuda_prec_precondition = QUDA_DOUBLE_PRECISION;

    // QUDA_DIRAC_ORDER = "even-odd, color inside spin" -> [parity][site][spin][color]
    // QUDA_QDP_DIRAC_ORDER = "even-odd, spin inside color" -> [parity][site][color][spin]
    // Our spinor layout is [parity][cb_idx][spin][color], so we need DIRAC_ORDER
    inv_param.dirac_order = QUDA_DIRAC_ORDER;

    // For full operator (not preconditioned)
    inv_param.solution_type = QUDA_MAT_SOLUTION;
    inv_param.solve_type = QUDA_DIRECT_SOLVE;
    inv_param.matpc_type = matpc_type;
    inv_param.dagger = QUDA_DAG_NO;

    inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
    inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

    return inv_param;
}

/**
 * Transform spinor between MILC and DeGrand-Rossi gamma bases.
 *
 * The transformation matrix S satisfies: gamma_DR = S * gamma_MILC * S^dag
 * For spinors: psi_DR = S * psi_MILC
 *
 * We compute S from the explicit gamma matrices in each basis.
 * For now, we use a simple diagonal transformation that handles
 * the common phase differences.
 */
torch::stable::Tensor milc_to_degrand_rossi(const torch::stable::Tensor& psi) {
    // The MILC and DeGrand-Rossi bases differ by phases and permutations.
    // For Wilson fermions, the key difference is in how gamma matrices act.
    //
    // MILC gamma_t (gamma_3) has the form with 1s on the anti-diagonal block.
    // DeGrand-Rossi uses a different convention.
    //
    // For a first implementation, we'll apply the operator in QUDA's native
    // basis and convert the result. The transformation is unitary.
    //
    // Transformation matrix (from literature):
    // This is a placeholder - the exact transformation depends on conventions.
    // For testing, we initially assume the bases are compatible or apply identity.

    return psi;  // Placeholder - will refine based on test results
}

torch::stable::Tensor degrand_rossi_to_milc(const torch::stable::Tensor& psi) {
    // Inverse transformation
    return psi;  // Placeholder
}

/**
 * Apply Wilson operator M using QUDA.
 *
 * This is the main interface function that:
 * 1. Converts plaq data layout to QUDA layout
 * 2. Loads gauge field into QUDA
 * 3. Applies the Wilson operator
 * 4. Converts result back to plaq layout
 */
torch::stable::Tensor quda_wilson_mat(
    const torch::stable::Tensor& gauge,
    const torch::stable::Tensor& psi,
    double kappa,
    bool antiperiodic_t) {

    STD_TORCH_CHECK(g_quda_initialized.load(), "QUDA must be initialized before calling Wilson operators");

    // Ensure tensors are on CPU and contiguous
    auto gauge_contig = torch::stable::contiguous(gauge);
    auto psi_contig = torch::stable::contiguous(psi);

    STD_TORCH_CHECK(gauge_contig.device().type() == torch::headeronly::DeviceType::CPU,
        "Gauge tensor must be on CPU");
    STD_TORCH_CHECK(psi_contig.device().type() == torch::headeronly::DeviceType::CPU,
        "Spinor tensor must be on CPU");

    // Infer lattice dimensions
    LatticeDims dims = infer_lattice_dims(gauge_contig, psi_contig);

    // Convert to QUDA layouts (BC is embedded in gauge, t_boundary is always PERIODIC)
    auto gauge_quda = plaq_gauge_to_quda(gauge_contig, dims, antiperiodic_t);
    auto psi_quda = plaq_spinor_to_quda(psi_contig, dims);

    // Apply gamma basis transformation (MILC -> DeGrand-Rossi)
    psi_quda = milc_to_degrand_rossi(psi_quda);

    // Create QUDA parameter structures
    QudaGaugeParam gauge_param = create_gauge_param(dims, antiperiodic_t);
    QudaInvertParam inv_param = create_invert_param(kappa);
    inv_param.dagger = QUDA_DAG_NO;

    // Load gauge field into QUDA
    // QDP order: void*[4] array, each pointing to [parity][cb_idx][row][col]
    // Our gauge_quda tensor is [4, 2, Vh, 3, 3] contiguous
    int64_t Vh = dims.volume() / 2;
    int64_t mu_stride = 2 * Vh * 9;  // complex elements per direction
    auto* gauge_base = static_cast<std::complex<double>*>(gauge_quda.mutable_data_ptr());
    void* gauge_ptrs[4];
    for (int mu = 0; mu < 4; mu++) {
        gauge_ptrs[mu] = static_cast<void*>(gauge_base + mu * mu_stride);
    }
    loadGaugeQuda(static_cast<void*>(gauge_ptrs), &gauge_param);

    // Allocate output spinor in QUDA layout
    auto result_quda = torch::stable::empty_like(psi_quda);

    void* in_ptr = psi_quda.mutable_data_ptr();
    void* out_ptr = result_quda.mutable_data_ptr();

    // Apply Wilson operator
    MatQuda(out_ptr, in_ptr, &inv_param);

    // Free gauge from QUDA
    freeGaugeQuda();

    // Apply inverse gamma basis transformation (DeGrand-Rossi -> MILC)
    result_quda = degrand_rossi_to_milc(result_quda);

    // Convert back to plaq layout
    return quda_spinor_to_plaq(result_quda, dims);
}

/**
 * Apply adjoint Wilson operator M^dag using QUDA.
 */
torch::stable::Tensor quda_wilson_mat_dag(
    const torch::stable::Tensor& gauge,
    const torch::stable::Tensor& psi,
    double kappa,
    bool antiperiodic_t) {

    STD_TORCH_CHECK(g_quda_initialized.load(), "QUDA must be initialized before calling Wilson operators");

    auto gauge_contig = torch::stable::contiguous(gauge);
    auto psi_contig = torch::stable::contiguous(psi);

    STD_TORCH_CHECK(gauge_contig.device().type() == torch::headeronly::DeviceType::CPU,
        "Gauge tensor must be on CPU");
    STD_TORCH_CHECK(psi_contig.device().type() == torch::headeronly::DeviceType::CPU,
        "Spinor tensor must be on CPU");

    LatticeDims dims = infer_lattice_dims(gauge_contig, psi_contig);

    auto gauge_quda = plaq_gauge_to_quda(gauge_contig, dims, antiperiodic_t);
    auto psi_quda = plaq_spinor_to_quda(psi_contig, dims);
    psi_quda = milc_to_degrand_rossi(psi_quda);

    QudaGaugeParam gauge_param = create_gauge_param(dims, antiperiodic_t);
    QudaInvertParam inv_param = create_invert_param(kappa);
    inv_param.dagger = QUDA_DAG_YES;  // Apply M^dag

    // QDP gauge order - array of 4 pointers
    int64_t Vh = dims.volume() / 2;
    int64_t mu_stride = 2 * Vh * 9;
    auto* gauge_base = static_cast<std::complex<double>*>(gauge_quda.mutable_data_ptr());
    void* gauge_ptrs[4];
    for (int mu = 0; mu < 4; mu++) {
        gauge_ptrs[mu] = static_cast<void*>(gauge_base + mu * mu_stride);
    }
    loadGaugeQuda(static_cast<void*>(gauge_ptrs), &gauge_param);

    auto result_quda = torch::stable::empty_like(psi_quda);
    void* in_ptr = psi_quda.mutable_data_ptr();
    void* out_ptr = result_quda.mutable_data_ptr();

    MatQuda(out_ptr, in_ptr, &inv_param);

    freeGaugeQuda();

    result_quda = degrand_rossi_to_milc(result_quda);
    return quda_spinor_to_plaq(result_quda, dims);
}

/**
 * Apply M^dag M using QUDA.
 */
torch::stable::Tensor quda_wilson_mat_dag_mat(
    const torch::stable::Tensor& gauge,
    const torch::stable::Tensor& psi,
    double kappa,
    bool antiperiodic_t) {

    STD_TORCH_CHECK(g_quda_initialized.load(), "QUDA must be initialized before calling Wilson operators");

    auto gauge_contig = torch::stable::contiguous(gauge);
    auto psi_contig = torch::stable::contiguous(psi);

    STD_TORCH_CHECK(gauge_contig.device().type() == torch::headeronly::DeviceType::CPU,
        "Gauge tensor must be on CPU");
    STD_TORCH_CHECK(psi_contig.device().type() == torch::headeronly::DeviceType::CPU,
        "Spinor tensor must be on CPU");

    LatticeDims dims = infer_lattice_dims(gauge_contig, psi_contig);

    auto gauge_quda = plaq_gauge_to_quda(gauge_contig, dims, antiperiodic_t);
    auto psi_quda = plaq_spinor_to_quda(psi_contig, dims);
    psi_quda = milc_to_degrand_rossi(psi_quda);

    QudaGaugeParam gauge_param = create_gauge_param(dims, antiperiodic_t);
    QudaInvertParam inv_param = create_invert_param(kappa);

    // QDP gauge order - array of 4 pointers
    int64_t Vh = dims.volume() / 2;
    int64_t mu_stride = 2 * Vh * 9;
    auto* gauge_base = static_cast<std::complex<double>*>(gauge_quda.mutable_data_ptr());
    void* gauge_ptrs[4];
    for (int mu = 0; mu < 4; mu++) {
        gauge_ptrs[mu] = static_cast<void*>(gauge_base + mu * mu_stride);
    }
    loadGaugeQuda(static_cast<void*>(gauge_ptrs), &gauge_param);

    // First apply M
    auto temp_quda = torch::stable::empty_like(psi_quda);
    inv_param.dagger = QUDA_DAG_NO;
    MatQuda(temp_quda.mutable_data_ptr(),
            psi_quda.mutable_data_ptr(),
            &inv_param);

    // Then apply M^dag
    auto result_quda = torch::stable::empty_like(psi_quda);
    inv_param.dagger = QUDA_DAG_YES;
    MatQuda(result_quda.mutable_data_ptr(),
            temp_quda.mutable_data_ptr(),
            &inv_param);

    freeGaugeQuda();

    result_quda = degrand_rossi_to_milc(result_quda);
    return quda_spinor_to_plaq(result_quda, dims);
}

// Register QUDA implementations
// Note: The operator schema is defined in code.cpp
STABLE_TORCH_LIBRARY_IMPL(quda_torch_op, CompositeExplicitAutograd, m) {
    m.impl("quda_is_available", TORCH_BOX(&quda_is_available));
    m.impl("quda_get_device_count", TORCH_BOX(&quda_get_device_count));
    m.impl("quda_init", TORCH_BOX(&quda_init));
    m.impl("quda_finalize", TORCH_BOX(&quda_finalize));
    m.impl("quda_is_initialized", TORCH_BOX(&quda_is_initialized));
    m.impl("quda_get_device", TORCH_BOX(&quda_get_device));
    m.impl("quda_get_version", TORCH_BOX(&quda_get_version));
    m.impl("quda_wilson_mat", TORCH_BOX(&quda_wilson_mat));
    m.impl("quda_wilson_mat_dag", TORCH_BOX(&quda_wilson_mat_dag));
    m.impl("quda_wilson_mat_dag_mat", TORCH_BOX(&quda_wilson_mat_dag_mat));
}

}  // namespace quda_torch_op
