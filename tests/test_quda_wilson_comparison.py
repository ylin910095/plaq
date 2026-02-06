"""Tests comparing QUDA and plaq Wilson Dirac operator implementations.

This module verifies that the QUDA backend produces identical results to
plaq's native Wilson operator implementation.

Tests are skipped if QUDA is not available.
"""

import pytest
import torch

import plaq as pq
from plaq.conventions.gamma_milc import get_gamma5

# Check if QUDA is available
try:
    import quda_torch_op

    QUDA_AVAILABLE = quda_torch_op.quda_is_available()
except ImportError:
    QUDA_AVAILABLE = False

pytestmark = pytest.mark.skipif(not QUDA_AVAILABLE, reason="QUDA not available")


@pytest.fixture(scope="module")
def quda_initialized():
    """Initialize QUDA for the test module and finalize on cleanup."""
    if QUDA_AVAILABLE:
        quda_torch_op.quda_init(0)
        yield
        quda_torch_op.quda_finalize()
    else:
        yield


def _apply_M_quda(
    U: pq.GaugeField,
    psi: pq.SpinorField,
    params: pq.WilsonParams,
    bc: pq.BoundaryCondition,
) -> pq.SpinorField:
    """Apply Wilson M operator using QUDA backend.

    Note: QUDA uses M = 1 - kappa*D normalization while plaq uses
    M = (m0 + 4r) - D/2 = 1/(2*kappa) - D/2.
    The relationship is: M_plaq = M_QUDA / (2*kappa).
    We rescale the QUDA result to match plaq's convention.
    """
    # Get raw tensors in plaq layout
    gauge_data = U.data  # [4, V, 3, 3]
    psi_data = psi.site  # [V, 4, 3]

    # Determine boundary condition
    antiperiodic_t = bc.get_bc_phase(3) < 0  # time direction

    # Apply QUDA operator
    result_data = quda_torch_op.quda_wilson_mat(gauge_data, psi_data, params.kappa, antiperiodic_t)

    # Rescale to match plaq convention: M_plaq = M_QUDA / (2*kappa)
    result_data = result_data / (2.0 * params.kappa)

    return pq.SpinorField(result_data, psi.lattice, layout="site")


def _apply_Mdag_quda(
    U: pq.GaugeField,
    psi: pq.SpinorField,
    params: pq.WilsonParams,
    bc: pq.BoundaryCondition,
) -> pq.SpinorField:
    """Apply Wilson M^dag operator using QUDA backend.

    Note: Same normalization rescaling as _apply_M_quda.
    """
    gauge_data = U.data
    psi_data = psi.site
    antiperiodic_t = bc.get_bc_phase(3) < 0

    result_data = quda_torch_op.quda_wilson_mat_dag(
        gauge_data, psi_data, params.kappa, antiperiodic_t
    )

    # Rescale to match plaq convention
    result_data = result_data / (2.0 * params.kappa)

    return pq.SpinorField(result_data, psi.lattice, layout="site")


def _apply_MdagM_quda(
    U: pq.GaugeField,
    psi: pq.SpinorField,
    params: pq.WilsonParams,
    bc: pq.BoundaryCondition,
) -> pq.SpinorField:
    """Apply Wilson M^dag M operator using QUDA backend.

    Note: For M^dag M, we need (M_QUDA / (2*kappa))^dag * (M_QUDA / (2*kappa))
    = M_QUDA^dag * M_QUDA / (2*kappa)^2
    """
    gauge_data = U.data
    psi_data = psi.site
    antiperiodic_t = bc.get_bc_phase(3) < 0

    result_data = quda_torch_op.quda_wilson_mat_dag_mat(
        gauge_data, psi_data, params.kappa, antiperiodic_t
    )

    # Rescale: (2*kappa)^2 for the product M^dag M
    result_data = result_data / (4.0 * params.kappa * params.kappa)

    return pq.SpinorField(result_data, psi.lattice, layout="site")


@pytest.mark.usefixtures("quda_initialized")
class TestApplyMComparison:
    """Tests comparing apply_M between QUDA and plaq."""

    def test_apply_M_identity_gauge(self) -> None:
        """Compare M with identity gauge field (free field case)."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        psi = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)

        # Apply with plaq
        result_plaq = pq.apply_M(U, psi, params, bc)

        # Apply with QUDA
        result_quda = _apply_M_quda(U, psi, params, bc)

        # Compare
        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        assert diff < 1e-10, f"M comparison failed for identity gauge: max diff = {diff}"

    def test_apply_M_random_gauge(self) -> None:
        """Compare M with random SU(3) gauge configuration."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        # Create a random gauge field (approximately SU(3))
        torch.manual_seed(42)
        U = pq.GaugeField.random(lat)
        psi = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)

        result_plaq = pq.apply_M(U, psi, params, bc)
        result_quda = _apply_M_quda(U, psi, params, bc)

        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        assert diff < 1e-10, f"M comparison failed for random gauge: max diff = {diff}"

    @pytest.mark.parametrize("mass", [0.0, 0.1, 0.5, 1.0])
    def test_apply_M_various_masses(self, mass: float) -> None:
        """Compare M for different mass values."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        psi = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=mass)

        result_plaq = pq.apply_M(U, psi, params, bc)
        result_quda = _apply_M_quda(U, psi, params, bc)

        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        assert diff < 1e-10, f"M comparison failed for mass={mass}: max diff = {diff}"


@pytest.mark.usefixtures("quda_initialized")
class TestApplyMdagComparison:
    """Tests comparing apply_Mdag between QUDA and plaq."""

    def test_apply_Mdag_identity_gauge(self) -> None:
        """Compare M^dag with identity gauge field."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        psi = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)

        result_plaq = pq.apply_Mdag(U, psi, params, bc)
        result_quda = _apply_Mdag_quda(U, psi, params, bc)

        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        assert diff < 1e-10, f"Mdag comparison failed: max diff = {diff}"

    def test_apply_Mdag_random_gauge(self) -> None:
        """Compare M^dag with random gauge configuration."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        torch.manual_seed(123)
        U = pq.GaugeField.random(lat)
        psi = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)

        result_plaq = pq.apply_Mdag(U, psi, params, bc)
        result_quda = _apply_Mdag_quda(U, psi, params, bc)

        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        assert diff < 1e-10, f"Mdag comparison failed for random gauge: max diff = {diff}"


@pytest.mark.usefixtures("quda_initialized")
class TestApplyMdagMComparison:
    """Tests comparing apply_MdagM between QUDA and plaq."""

    def test_apply_MdagM_identity_gauge(self) -> None:
        """Compare M^dag M with identity gauge field."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        psi = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)

        result_plaq = pq.apply_MdagM(U, psi, params, bc)
        result_quda = _apply_MdagM_quda(U, psi, params, bc)

        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        assert diff < 1e-10, f"MdagM comparison failed: max diff = {diff}"

    def test_apply_MdagM_random_gauge(self) -> None:
        """Compare M^dag M with random gauge configuration."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        torch.manual_seed(456)
        U = pq.GaugeField.random(lat)
        psi = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)

        result_plaq = pq.apply_MdagM(U, psi, params, bc)
        result_quda = _apply_MdagM_quda(U, psi, params, bc)

        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        # Slightly relaxed tolerance for MdagM (two operator applications)
        assert diff < 1e-9, f"MdagM comparison failed for random gauge: max diff = {diff}"


@pytest.mark.usefixtures("quda_initialized")
class TestBoundaryConditions:
    """Tests comparing boundary condition handling."""

    def test_antiperiodic_bc(self) -> None:
        """Compare with antiperiodic BC in time (standard for fermions)."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition(fermion_bc_time=-1.0)  # Antiperiodic
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        psi = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)

        result_plaq = pq.apply_M(U, psi, params, bc)
        result_quda = _apply_M_quda(U, psi, params, bc)

        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        assert diff < 1e-10, f"Antiperiodic BC comparison failed: max diff = {diff}"

    def test_periodic_bc(self) -> None:
        """Compare with periodic BC in time."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition(fermion_bc_time=1.0)  # Periodic
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        psi = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)

        result_plaq = pq.apply_M(U, psi, params, bc)
        result_quda = _apply_M_quda(U, psi, params, bc)

        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        assert diff < 1e-10, f"Periodic BC comparison failed: max diff = {diff}"

    def test_bc_affects_result(self) -> None:
        """Verify that changing BC changes the result (sanity check)."""
        lat = pq.Lattice((4, 4, 4, 4))

        U = pq.GaugeField.eye(lat)
        torch.manual_seed(789)
        psi = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)

        # Antiperiodic BC
        bc_anti = pq.BoundaryCondition(fermion_bc_time=-1.0)
        lat.build_neighbor_tables(bc_anti)
        result_anti = _apply_M_quda(U, psi, params, bc_anti)

        # Periodic BC
        bc_periodic = pq.BoundaryCondition(fermion_bc_time=1.0)
        lat.build_neighbor_tables(bc_periodic)
        result_periodic = _apply_M_quda(U, psi, params, bc_periodic)

        diff = torch.abs(result_anti.site - result_periodic.site).max().item()
        assert diff > 1e-10, "BC change should affect the result"


@pytest.mark.usefixtures("quda_initialized")
class TestQudaGamma5Hermiticity:
    """Tests verifying QUDA satisfies gamma5-hermiticity."""

    def test_gamma5_hermiticity_quda(self) -> None:
        """Verify QUDA operator satisfies M^dag = gamma5 M gamma5."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        psi = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)

        # Get gamma5
        gamma5 = get_gamma5(dtype=psi.dtype, device=psi.device)

        # Compute M^dag psi using QUDA
        Mdag_psi = _apply_Mdag_quda(U, psi, params, bc)

        # Compute gamma5 M gamma5 psi using QUDA
        psi_site = psi.site
        g5_psi = torch.einsum("ab,vbc->vac", gamma5, psi_site)
        g5_psi_field = pq.SpinorField(g5_psi, lat)

        M_g5_psi = _apply_M_quda(U, g5_psi_field, params, bc)
        g5_M_g5_psi = torch.einsum("ab,vbc->vac", gamma5, M_g5_psi.site)

        # Compare
        diff = torch.abs(Mdag_psi.site - g5_M_g5_psi).max().item()
        assert diff < 1e-10, f"QUDA gamma5-hermiticity failed: max diff = {diff}"

    def test_adjoint_inner_product_quda(self) -> None:
        """Verify <phi, M psi> = <M^dag phi, psi> for QUDA operator."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        torch.manual_seed(101)
        psi = pq.SpinorField.random(lat)
        phi = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)

        # <phi, M psi>
        M_psi = _apply_M_quda(U, psi, params, bc)
        lhs = torch.vdot(phi.site.flatten(), M_psi.site.flatten())

        # <M^dag phi, psi>
        Mdag_phi = _apply_Mdag_quda(U, phi, params, bc)
        rhs = torch.vdot(Mdag_phi.site.flatten(), psi.site.flatten())

        diff = torch.abs(lhs - rhs).item()
        assert diff < 1e-10, f"QUDA adjoint test failed: |lhs - rhs| = {diff}"


@pytest.mark.usefixtures("quda_initialized")
class TestDiagnosticSingleSite:
    """Diagnostic tests using single-site spinors to isolate convention issues."""

    def test_single_site_constant_spinor(self) -> None:
        """Test with constant spinor at spin=0, color=0."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition(fermion_bc_time=1.0)  # Periodic for simplicity
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)

        # Constant spinor
        psi_data = torch.zeros(lat.volume, 4, 3, dtype=torch.complex128)
        psi_data[:, 0, 0] = 1.0
        psi = pq.SpinorField(psi_data, lat)

        params = pq.WilsonParams(mass=0.1)

        result_plaq = pq.apply_M(U, psi, params, bc)
        result_quda = _apply_M_quda(U, psi, params, bc)

        # For constant spinor with periodic BC, result should be m0 * psi
        # Check both give same result
        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        assert diff < 1e-10, f"Single site test failed: max diff = {diff}"

        # Also verify the eigenvalue is m0
        expected_eigenvalue = params.mass
        result_val = result_quda.site[0, 0, 0].real.item()
        assert abs(result_val - expected_eigenvalue) < 1e-10, (
            f"Eigenvalue mismatch: got {result_val}, expected {expected_eigenvalue}"
        )

    def test_single_site_isolated_spinor(self) -> None:
        """Test with spinor non-zero at only one site."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)

        # Single site spinor at origin
        psi_data = torch.zeros(lat.volume, 4, 3, dtype=torch.complex128)
        site_idx = lat.index(0, 0, 0, 0)  # Origin
        psi_data[site_idx, 0, 0] = 1.0
        psi = pq.SpinorField(psi_data, lat)

        params = pq.WilsonParams(mass=0.1)

        result_plaq = pq.apply_M(U, psi, params, bc)
        result_quda = _apply_M_quda(U, psi, params, bc)

        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        assert diff < 1e-10, f"Isolated spinor test failed: max diff = {diff}"


@pytest.mark.usefixtures("quda_initialized")
class TestLatticeSizes:
    """Tests with different lattice sizes."""

    @pytest.mark.parametrize(
        "lattice_shape",
        [
            (4, 4, 4, 4),
            (4, 4, 4, 8),
            (8, 8, 8, 8),
        ],
    )
    def test_various_lattice_sizes(self, lattice_shape: tuple[int, int, int, int]) -> None:
        """Compare operators on different lattice sizes."""
        lat = pq.Lattice(lattice_shape)
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        psi = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)

        result_plaq = pq.apply_M(U, psi, params, bc)
        result_quda = _apply_M_quda(U, psi, params, bc)

        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        assert diff < 1e-10, f"Comparison failed for lattice {lattice_shape}: max diff = {diff}"


@pytest.mark.usefixtures("quda_initialized")
class TestBoundaryHoppingDiagnostics:
    """Diagnostic tests for boundary condition handling."""

    def test_boundary_hopping_forward_t(self) -> None:
        """Test spinor propagation across t-boundary (forward hop t=Nt-1 -> t=0)."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition(fermion_bc_time=-1.0)  # Antiperiodic
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        params = pq.WilsonParams(mass=0.1)

        # Create spinor non-zero only at t=Nt-1, x=y=z=0
        psi_data = torch.zeros(lat.volume, 4, 3, dtype=torch.complex128)
        site_t3 = lat.index(0, 0, 0, 3)  # t=3 (Nt-1)
        psi_data[site_t3, 0, 0] = 1.0
        psi = pq.SpinorField(psi_data, lat)

        # Apply M - forward hopping from t=3 will wrap to t=0 with -1 phase
        result_plaq = pq.apply_M(U, psi, params, bc)
        result_quda = _apply_M_quda(U, psi, params, bc)

        # Check overall agreement
        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        assert diff < 1e-10, f"Boundary hopping forward test failed: max diff = {diff}"

    def test_boundary_hopping_backward_t(self) -> None:
        """Test spinor propagation across t-boundary (backward hop t=0 -> t=Nt-1)."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition(fermion_bc_time=-1.0)  # Antiperiodic
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        params = pq.WilsonParams(mass=0.1)

        # Create spinor non-zero only at t=0, x=y=z=0
        psi_data = torch.zeros(lat.volume, 4, 3, dtype=torch.complex128)
        site_t0 = lat.index(0, 0, 0, 0)  # t=0
        psi_data[site_t0, 0, 0] = 1.0
        psi = pq.SpinorField(psi_data, lat)

        # Apply M - backward hopping from t=0 will wrap to t=3 with -1 phase
        result_plaq = pq.apply_M(U, psi, params, bc)
        result_quda = _apply_M_quda(U, psi, params, bc)

        # Check overall agreement
        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        assert diff < 1e-10, f"Boundary hopping backward test failed: max diff = {diff}"

    def test_per_timeslice_comparison(self) -> None:
        """Compare results per timeslice to identify where mismatches occur."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition(fermion_bc_time=-1.0)  # Antiperiodic
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        torch.manual_seed(999)
        psi = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)

        result_plaq = pq.apply_M(U, psi, params, bc)
        result_quda = _apply_M_quda(U, psi, params, bc)

        # Check per-timeslice difference
        Nt = lat.shape[3]

        for t in range(Nt):
            # Get sites at this timeslice
            t_mask = torch.zeros(lat.volume, dtype=torch.bool)
            for site in range(lat.volume):
                coords = lat.coord(site)
                if coords[3] == t:
                    t_mask[site] = True

            diff_t = torch.abs(result_plaq.site[t_mask] - result_quda.site[t_mask]).max().item()
            assert diff_t < 1e-10, f"Timeslice t={t} mismatch: max diff = {diff_t}"

    def test_unique_value_antiperiodic(self) -> None:
        """Test with unique values at each site and antiperiodic BC."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition(fermion_bc_time=-1.0)  # Antiperiodic
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        params = pq.WilsonParams(mass=0.1)

        # Create spinor with unique value at each site (for site ordering verification)
        psi_data = torch.zeros(lat.volume, 4, 3, dtype=torch.complex128)
        for site in range(lat.volume):
            x, y, z, t = lat.coord(site)
            # Use complex number encoding coordinates
            psi_data[site, 0, 0] = complex(site + 1, x * 1000 + y * 100 + z * 10 + t)
        psi = pq.SpinorField(psi_data, lat)

        result_plaq = pq.apply_M(U, psi, params, bc)
        result_quda = _apply_M_quda(U, psi, params, bc)

        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        assert diff < 1e-10, f"Unique value antiperiodic test failed: max diff = {diff}"


@pytest.mark.usefixtures("quda_initialized")
class TestCbIdxFormula:
    """Tests to verify the checkerboard index formula used in QUDA conversion."""

    def test_cb_idx_formula_4x4x4x4(self) -> None:
        """Verify cb_idx = plaq_site >> 1 equals QUDA's explicit formula."""
        Nx, Ny, Nz, Nt = 4, 4, 4, 4
        Nxh = Nx // 2

        mismatches = []
        for t in range(Nt):
            for z in range(Nz):
                for y in range(Ny):
                    for x in range(Nx):
                        plaq_site = x + Nx * (y + Ny * (z + Nz * t))
                        parity = (x + y + z + t) % 2

                        # Method 1: bit shift (current code)
                        cb_v1 = plaq_site >> 1

                        # Method 2: QUDA explicit formula
                        cb_v2 = (x // 2) + Nxh * (y + Ny * (z + Nz * t))

                        if cb_v1 != cb_v2:
                            mismatches.append(
                                f"({x},{y},{z},{t}): plaq_site={plaq_site}, "
                                f"parity={parity}, cb_v1={cb_v1}, cb_v2={cb_v2}"
                            )

        assert len(mismatches) == 0, "cb_idx mismatches:\n" + "\n".join(mismatches[:20])

    def test_cb_idx_formula_2x2x2x2(self) -> None:
        """Verify cb_idx formula on 2x2x2x2 lattice (minimal for manual verification)."""
        Nx, Ny, Nz, Nt = 2, 2, 2, 2
        Nxh = Nx // 2

        print("\n2x2x2x2 site mapping:")
        print("site | (x,y,z,t) | parity | cb_shift | cb_quda")
        print("-" * 55)

        mismatches = []
        for t in range(Nt):
            for z in range(Nz):
                for y in range(Ny):
                    for x in range(Nx):
                        plaq_site = x + Nx * (y + Ny * (z + Nz * t))
                        parity = (x + y + z + t) % 2
                        cb_shift = plaq_site >> 1
                        cb_quda = (x // 2) + Nxh * (y + Ny * (z + Nz * t))

                        print(
                            f"{plaq_site:4d} | ({x},{y},{z},{t})     | "
                            f"{parity}      | {cb_shift:8d} | {cb_quda}"
                        )

                        if cb_shift != cb_quda:
                            mismatches.append((x, y, z, t, plaq_site, parity, cb_shift, cb_quda))

        if mismatches:
            print(f"\nMismatches found: {len(mismatches)}")
            for m in mismatches:
                print(
                    f"  ({m[0]},{m[1]},{m[2]},{m[3]}): site={m[4]}, parity={m[5]}, "
                    f"shift={m[6]}, quda={m[7]}"
                )

        assert len(mismatches) == 0, "cb_idx mismatches found"


@pytest.mark.usefixtures("quda_initialized")
class TestDetailedComparison:
    """Detailed per-site comparison tests for debugging."""

    def test_per_site_comparison_identity_gauge(self) -> None:
        """Compare per-site values with identity gauge to identify specific issues."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition(fermion_bc_time=1.0)  # Periodic for simplicity
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)

        # Single site spinor at origin
        psi_data = torch.zeros(lat.volume, 4, 3, dtype=torch.complex128)
        psi_data[0, 0, 0] = 1.0  # Site 0, spin 0, color 0
        psi = pq.SpinorField(psi_data, lat)

        params = pq.WilsonParams(mass=10.0)  # Large mass to emphasize diagonal

        result_plaq = pq.apply_M(U, psi, params, bc)
        result_quda = _apply_M_quda(U, psi, params, bc)

        # Print detailed comparison for non-zero sites
        print("\nPer-site comparison (non-zero values only):")
        print("site | (x,y,z,t) | spin | color | plaq         | quda         | diff")
        print("-" * 80)

        for site in range(min(lat.volume, 256)):
            for spin in range(4):
                for color in range(3):
                    val_plaq = result_plaq.site[site, spin, color]
                    val_quda = result_quda.site[site, spin, color]
                    if abs(val_plaq) > 1e-12 or abs(val_quda) > 1e-12:
                        diff = abs(val_plaq - val_quda)
                        coords = lat.coord(site)
                        print(
                            f"{site:4d} | {coords} | {spin}    | {color}     | "
                            f"{val_plaq.real:+.6f}{val_plaq.imag:+.6f}j | "
                            f"{val_quda.real:+.6f}{val_quda.imag:+.6f}j | {diff:.2e}"
                        )

        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        assert diff < 1e-10, f"Per-site comparison failed: max diff = {diff}"

    def test_forward_neighbor_only(self) -> None:
        """Test with spinor that should only affect forward neighbors."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition(fermion_bc_time=1.0)  # Periodic
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)

        # Spinor at site 0 (origin = (0,0,0,0))
        psi_data = torch.zeros(lat.volume, 4, 3, dtype=torch.complex128)
        psi_data[0, 0, 0] = 1.0
        psi = pq.SpinorField(psi_data, lat)

        params = pq.WilsonParams(mass=0.0)  # Zero mass - only hopping terms

        result_plaq = pq.apply_M(U, psi, params, bc)
        result_quda = _apply_M_quda(U, psi, params, bc)

        # Site 0's neighbors:
        # Forward: x+: site 1, y+: site 4, z+: site 16, t+: site 64
        # Backward: x-: site 3, y-: site 12, z-: site 48, t-: site 192

        print("\nAnalyzing hopping from site 0:")
        neighbor_sites = [
            (0, "source"),
            (1, "x+ forward"),
            (3, "x- backward (periodic wrap)"),
            (4, "y+ forward"),
            (12, "y- backward (periodic wrap)"),
            (16, "z+ forward"),
            (48, "z- backward (periodic wrap)"),
            (64, "t+ forward"),
            (192, "t- backward (periodic wrap)"),
        ]

        print(
            "\nsite | description          | plaq                  | quda                  | diff"
        )
        print("-" * 100)

        for site, desc in neighbor_sites:
            for spin in range(4):
                for color in range(3):
                    val_plaq = result_plaq.site[site, spin, color]
                    val_quda = result_quda.site[site, spin, color]
                    if abs(val_plaq) > 1e-12 or abs(val_quda) > 1e-12:
                        diff = abs(val_plaq - val_quda)
                        marker = "***" if diff > 1e-10 else ""
                        print(
                            f"{site:4d} | {desc:20} | s{spin}c{color} | "
                            f"{val_plaq.real:+.6f}{val_plaq.imag:+.6f}j | "
                            f"{val_quda.real:+.6f}{val_quda.imag:+.6f}j | {diff:.2e} {marker}"
                        )

        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        assert diff < 1e-10, f"Forward neighbor test failed: max diff = {diff}"

    def test_gauge_parity_association(self) -> None:
        """Test which parity stores links FROM vs TO even/odd sites."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition(fermion_bc_time=1.0)
        lat.build_neighbor_tables(bc)

        # Create gauge with special value at specific link
        # U_x(site 0) = diag(2, 1, 1) - should affect backward hop from site 1
        U_data = torch.eye(3, dtype=torch.complex128).unsqueeze(0).unsqueeze(0)
        U_data = U_data.expand(4, lat.volume, 3, 3).clone()
        U_data[0, 0, 0, 0] = 2.0  # U_x(0)[0,0] = 2

        U = pq.GaugeField(U_data, lat)

        # Spinor at site 1 (x=1, y=z=t=0, parity=1=odd)
        psi_data = torch.zeros(lat.volume, 4, 3, dtype=torch.complex128)
        psi_data[1, 0, 0] = 1.0  # color=0
        psi = pq.SpinorField(psi_data, lat)

        params = pq.WilsonParams(mass=0.0)

        result_plaq = pq.apply_M(U, psi, params, bc)
        result_quda = _apply_M_quda(U, psi, params, bc)

        print("\nGauge parity test:")
        print("U_x(site 0)[0,0] = 2 (other elements = identity)")
        print("Spinor at site 1 (odd parity), spin=0, color=0")
        print("Backward x-hop from site 1 uses U_x(0)^dag")
        print("\nChecking site 0 (should have factor of 2 in color=0):")

        for spin in range(4):
            for color in range(3):
                val_plaq = result_plaq.site[0, spin, color]
                val_quda = result_quda.site[0, spin, color]
                if abs(val_plaq) > 1e-12 or abs(val_quda) > 1e-12:
                    diff = abs(val_plaq - val_quda)
                    print(
                        f"  site=0, spin={spin}, color={color}: "
                        f"plaq={val_plaq:.6f}, quda={val_quda:.6f}, diff={diff:.2e}"
                    )

        print("\nAll non-zero values:")
        for site in range(lat.volume):
            for spin in range(4):
                for color in range(3):
                    val_plaq = result_plaq.site[site, spin, color]
                    val_quda = result_quda.site[site, spin, color]
                    if abs(val_plaq) > 1e-12 or abs(val_quda) > 1e-12:
                        diff = abs(val_plaq - val_quda)
                        coords = lat.coord(site)
                        marker = "***" if diff > 1e-10 else ""
                        print(
                            f"  site={site} {coords}, s{spin}c{color}: "
                            f"plaq={val_plaq:.6f}, quda={val_quda:.6f}, diff={diff:.2e} {marker}"
                        )

        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        assert diff < 1e-10, f"Gauge parity test failed: max diff = {diff}"

    def test_gauge_row_col_transpose(self) -> None:
        """Test if QUDA expects U^T instead of U for gauge matrices."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition(fermion_bc_time=1.0)
        lat.build_neighbor_tables(bc)

        # Create non-symmetric gauge: U_x(0)[0,1] = 1, U_x(0)[1,0] = 0
        U_data = torch.eye(3, dtype=torch.complex128).unsqueeze(0).unsqueeze(0)
        U_data = U_data.expand(4, lat.volume, 3, 3).clone()
        U_data[0, 0, 0, 1] = 1.0  # U[row=0, col=1] = 1

        U = pq.GaugeField(U_data, lat)

        # Spinor at site 1 with color=1
        psi_data = torch.zeros(lat.volume, 4, 3, dtype=torch.complex128)
        psi_data[1, 0, 1] = 1.0  # color=1
        psi = pq.SpinorField(psi_data, lat)

        params = pq.WilsonParams(mass=0.0)

        result_plaq = pq.apply_M(U, psi, params, bc)
        result_quda = _apply_M_quda(U, psi, params, bc)

        print("\nGauge row/col test:")
        print("U_x(0) has off-diagonal: U[0,1] = 1 (row=0, col=1)")
        print("Spinor at site 1, spin=0, color=1")
        print("If U @ psi: result at site 0 should have color=0 contribution (U[0,1]*psi[1])")
        print("If U^T @ psi: result at site 0 should have color=2 contribution (U^T[2,1]=0)")
        print("\nResult at site 0:")

        for spin in range(4):
            for color in range(3):
                val_plaq = result_plaq.site[0, spin, color]
                val_quda = result_quda.site[0, spin, color]
                if abs(val_plaq) > 1e-12 or abs(val_quda) > 1e-12:
                    diff = abs(val_plaq - val_quda)
                    print(
                        f"  spin={spin}, color={color}: "
                        f"plaq={val_plaq:.6f}, quda={val_quda:.6f}, diff={diff:.2e}"
                    )

        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        assert diff < 1e-10, f"Row/col transpose test failed: max diff = {diff}"

    def test_gauge_parity_swap_hypothesis(self) -> None:
        """Test if QUDA expects swapped parity for gauge fields."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition(fermion_bc_time=1.0)
        lat.build_neighbor_tables(bc)

        # Create gauge where even-site links differ from odd-site links
        # Even sites: U = diag(1, 1, 1)
        # Odd sites: U = diag(2, 2, 2)
        U_data = torch.eye(3, dtype=torch.complex128).unsqueeze(0).unsqueeze(0)
        U_data = U_data.expand(4, lat.volume, 3, 3).clone()

        for site in range(lat.volume):
            coords = lat.coord(site)
            parity = (coords[0] + coords[1] + coords[2] + coords[3]) % 2
            if parity == 1:  # Odd site
                U_data[:, site, :, :] = 2.0 * torch.eye(3, dtype=torch.complex128)

        U = pq.GaugeField(U_data, lat)

        # Spinor at even site 0
        psi_data = torch.zeros(lat.volume, 4, 3, dtype=torch.complex128)
        psi_data[0, 0, 0] = 1.0
        psi = pq.SpinorField(psi_data, lat)

        params = pq.WilsonParams(mass=0.0)

        result_plaq = pq.apply_M(U, psi, params, bc)
        result_quda = _apply_M_quda(U, psi, params, bc)

        print("\nGauge parity swap test:")
        print("Even-site links: U = I")
        print("Odd-site links: U = 2*I")
        print("Spinor at site 0 (even), spin=0, color=0")
        print("\nForward hops from site 0 use U(site 0) which should be I")
        print("Backward hops use U^dag from neighbor, which is odd-parity so 2*I")

        neighbor_sites = [
            (0, "source (even)"),
            (1, "x+ forward (odd) - uses U_x(0)=I"),
            (3, "x- backward (odd) - uses U_x(3)^dag=2I"),
            (4, "y+ forward (odd) - uses U_y(0)=I"),
            (12, "y- backward (odd) - uses U_y(12)^dag=2I"),
        ]

        print("\nsite | description | spin | color | plaq | quda | diff")
        print("-" * 80)

        for site, desc in neighbor_sites:
            for spin in range(4):
                for color in range(3):
                    val_plaq = result_plaq.site[site, spin, color]
                    val_quda = result_quda.site[site, spin, color]
                    if abs(val_plaq) > 1e-12 or abs(val_quda) > 1e-12:
                        diff = abs(val_plaq - val_quda)
                        marker = "***" if diff > 1e-10 else ""
                        print(
                            f"{site:4d} | {desc:40} | s{spin}c{color} | "
                            f"{val_plaq.real:+.4f} | {val_quda.real:+.4f} | {diff:.2e} {marker}"
                        )

        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        assert diff < 1e-10, f"Parity swap test failed: max diff = {diff}"

    def test_gauge_link_identification(self) -> None:
        """Test which specific gauge link QUDA is reading."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition(fermion_bc_time=1.0)
        lat.build_neighbor_tables(bc)

        # Create gauge where each site has unique diagonal element
        # U_x(site) = diag(site+1, 1, 1)
        U_data = torch.eye(3, dtype=torch.complex128).unsqueeze(0).unsqueeze(0)
        U_data = U_data.expand(4, lat.volume, 3, 3).clone()

        for mu in range(4):
            for site in range(lat.volume):
                U_data[mu, site, 0, 0] = float(site + 1)

        U = pq.GaugeField(U_data, lat)

        # Spinor at site 0, color=0
        psi_data = torch.zeros(lat.volume, 4, 3, dtype=torch.complex128)
        psi_data[0, 0, 0] = 1.0
        psi = pq.SpinorField(psi_data, lat)

        params = pq.WilsonParams(mass=0.0)

        result_plaq = pq.apply_M(U, psi, params, bc)
        result_quda = _apply_M_quda(U, psi, params, bc)

        print("\nGauge link identification test:")
        print("U_mu(site)[0,0] = site+1, other diagonal = 1")
        print("Spinor at site 0, spin=0, color=0")
        print("\nForward x-hop: site 0 -> site 1, uses U_x(0)[0,0] = 1")
        print("Backward x-hop: site 0 -> site 3, uses U_x(3)^dag[0,0] = 4")
        print("\nIf gauge parity is swapped, forward x-hop would use U_x(1)=2")
        print("and backward x-hop would use U_x(2)=3")

        # site 0's neighbors:
        # Forward x: site 1 (uses U_x(0))
        # Backward x: site 3 (uses U_x(3)^dag)
        neighbor_info = [
            (1, "x+ fwd (should use U_x(0)=1)", "fwd"),
            (3, "x- bwd (should use U_x(3)=4)", "bwd"),
            (4, "y+ fwd (should use U_y(0)=1)", "fwd"),
            (12, "y- bwd (should use U_y(12)=13)", "bwd"),
        ]

        print("\nsite | description | s0c0 plaq | s0c0 quda | inferred gauge | expected")
        print("-" * 90)

        for site, desc, hop_type in neighbor_info:
            val_plaq = result_plaq.site[site, 0, 0].real
            val_quda = result_quda.site[site, 0, 0].real

            # The hopping term contributes -0.5 * U * (1 +/- gamma) * psi
            # For spin=0, color=0 with identity-like gauge, result is -0.5 * U[0,0]
            # So the gauge element is approximately -2 * result
            inferred_plaq = abs(val_plaq) * 2
            inferred_quda = abs(val_quda) * 2

            if hop_type == "fwd":
                expected = {"1 x+": 1, "4 y+": 1, "16 z+": 1, "64 t+": 1}.get(
                    f"{site} {desc[:2]}", "?"
                )
            else:
                expected = {"3 x-": 4, "12 y-": 13, "48 z-": 49, "192 t-": 193}.get(
                    f"{site} {desc[:2]}", "?"
                )

            print(
                f"{site:4d} | {desc:40} | {val_plaq:+.4f} | {val_quda:+.4f} | "
                f"plaq~{inferred_plaq:.0f} quda~{inferred_quda:.0f} | expected~{expected}"
            )

        # Let's also look at what QUDA produces at site 0
        print("\nAt source site 0:")
        for spin in range(4):
            for color in range(3):
                val_plaq = result_plaq.site[0, spin, color]
                val_quda = result_quda.site[0, spin, color]
                if abs(val_plaq) > 1e-12 or abs(val_quda) > 1e-12:
                    print(f"  s{spin}c{color}: plaq={val_plaq:.6f}, quda={val_quda:.6f}")

        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        assert diff < 1e-10, f"Gauge link identification test failed: max diff = {diff}"

    def test_trace_backward_gauge_access(self) -> None:
        """Trace exactly which gauge link QUDA reads for backward hops."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition(fermion_bc_time=1.0)
        lat.build_neighbor_tables(bc)

        # Create gauge where each (parity, cb_idx) has unique value
        # This helps us identify which gauge slot QUDA reads from
        U_data = torch.eye(3, dtype=torch.complex128).unsqueeze(0).unsqueeze(0)
        U_data = U_data.expand(4, lat.volume, 3, 3).clone()

        for site in range(lat.volume):
            coords = lat.coord(site)
            parity = (coords[0] + coords[1] + coords[2] + coords[3]) % 2
            cb_idx = site >> 1
            # Encode (parity, cb_idx) into gauge value
            # Value = 1 + parity * 1000 + cb_idx
            U_data[:, site, 0, 0] = 1.0 + parity * 1000 + cb_idx

        U = pq.GaugeField(U_data, lat)

        # Test backward x-hop from site 0 (even, parity=0, cb_idx=0)
        # Target is site 3 (odd, parity=1, cb_idx=1)
        # Plaq expects to use U_x(3) which has value 1 + 1*1000 + 1 = 1002

        psi_data = torch.zeros(lat.volume, 4, 3, dtype=torch.complex128)
        psi_data[0, 0, 0] = 1.0  # At site 0
        psi = pq.SpinorField(psi_data, lat)

        params = pq.WilsonParams(mass=0.0)

        result_plaq = pq.apply_M(U, psi, params, bc)
        result_quda = _apply_M_quda(U, psi, params, bc)

        print("\nBackward gauge access trace:")
        print("Source site 0: (0,0,0,0), parity=0, cb_idx=0")
        print(f"Gauge U_x(0) value: {U_data[0, 0, 0, 0].real:.0f} (encoded: parity=0, cb=0)")
        print()

        neighbor_info = [
            (3, "x-", "U_x(3)", (3, 0, 0, 0)),
            (12, "y-", "U_y(12)", (0, 3, 0, 0)),
            (48, "z-", "U_z(48)", (0, 0, 3, 0)),
            (192, "t-", "U_t(192)", (0, 0, 0, 3)),
        ]

        print("Backward hop analysis:")
        print("-" * 80)
        for target_site, dir_name, gauge_name, coords in neighbor_info:
            parity = sum(coords) % 2
            cb_idx = target_site >> 1
            expected_gauge_val = 1.0 + parity * 1000 + cb_idx
            actual_gauge_val = U_data[
                0 if "x" in dir_name else 1 if "y" in dir_name else 2 if "z" in dir_name else 3,
                target_site,
                0,
                0,
            ].real

            # The result at target_site should be ~ -0.5 * gauge_val (for spin=0, color=0)
            val_plaq = result_plaq.site[target_site, 0, 0].real
            val_quda = result_quda.site[target_site, 0, 0].real

            inferred_plaq_gauge = abs(val_plaq) * 2
            inferred_quda_gauge = abs(val_quda) * 2

            print(
                f"{dir_name}: target site {target_site} {coords}, parity={parity}, cb_idx={cb_idx}"
            )
            print(f"     Expected gauge {gauge_name} val: {expected_gauge_val:.0f}")
            print(f"     Actual gauge val at site: {actual_gauge_val:.0f}")
            print(f"     Plaq result: {val_plaq:.4f} -> inferred gauge ~{inferred_plaq_gauge:.0f}")
            print(f"     QUDA result: {val_quda:.4f} -> inferred gauge ~{inferred_quda_gauge:.0f}")

            # Decode what gauge slot QUDA read from
            if inferred_quda_gauge > 500:
                quda_parity = int(inferred_quda_gauge) // 1000
                quda_cb = (int(inferred_quda_gauge) - 1) % 1000
                print(f"     QUDA seems to read: parity={quda_parity}, cb_idx={quda_cb}")
            else:
                print(f"     QUDA read value ~{inferred_quda_gauge:.0f}")
            print()

        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        # Don't assert, just report
        print(f"Max diff: {diff}")

    def test_mu_direction_ordering(self) -> None:
        """Test if QUDA expects different mu ordering (x,y,z,t vs t,z,y,x)."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition(fermion_bc_time=1.0)
        lat.build_neighbor_tables(bc)

        # Create gauge where mu=0 has value 10, mu=1 has 20, etc.
        U_data = torch.eye(3, dtype=torch.complex128).unsqueeze(0).unsqueeze(0)
        U_data = U_data.expand(4, lat.volume, 3, 3).clone()

        for mu in range(4):
            U_data[mu, :, 0, 0] = (mu + 1) * 10.0

        U = pq.GaugeField(U_data, lat)

        # Test from site 0 - forward hops go to sites 1,4,16,64 for x,y,z,t
        psi_data = torch.zeros(lat.volume, 4, 3, dtype=torch.complex128)
        psi_data[0, 0, 0] = 1.0
        psi = pq.SpinorField(psi_data, lat)

        params = pq.WilsonParams(mass=0.0)

        result_plaq = pq.apply_M(U, psi, params, bc)
        result_quda = _apply_M_quda(U, psi, params, bc)

        print("\nMu direction ordering test:")
        print("Gauge values: mu=0 (x) -> 10, mu=1 (y) -> 20, mu=2 (z) -> 30, mu=3 (t) -> 40")
        print()

        forward_sites = [
            (1, "x+ (mu=0)", 10),
            (4, "y+ (mu=1)", 20),
            (16, "z+ (mu=2)", 30),
            (64, "t+ (mu=3)", 40),
        ]

        print("Forward hops (uses U at source site 0):")
        print("-" * 60)
        for site, desc, expected_gauge in forward_sites:
            val_plaq = result_plaq.site[site, 0, 0].real
            val_quda = result_quda.site[site, 0, 0].real
            inferred_plaq = abs(val_plaq) * 2
            inferred_quda = abs(val_quda) * 2
            print(
                f"Site {site:3d} {desc}: plaq~{inferred_plaq:.0f}, quda~{inferred_quda:.0f}, expected={expected_gauge}"
            )

        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        print(f"\nMax diff: {diff}")

    def test_full_gauge_tracing(self) -> None:
        """Trace exactly which (mu, parity, cb_idx) QUDA reads for each hop."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition(fermion_bc_time=1.0)
        lat.build_neighbor_tables(bc)

        # Test with identity gauge first - all hops should give same value
        # This tests if the Dslash structure is correct independent of gauge direction
        U_data = torch.eye(3, dtype=torch.complex128).unsqueeze(0).unsqueeze(0)
        U_data = U_data.expand(4, lat.volume, 3, 3).clone()

        # Make gauge identity for all directions
        # All forward/backward contributions should be identical then
        U = pq.GaugeField(U_data, lat)

        psi_data = torch.zeros(lat.volume, 4, 3, dtype=torch.complex128)
        psi_data[0, 0, 0] = 1.0  # At site 0 (parity=0, cb_idx=0)
        psi = pq.SpinorField(psi_data, lat)

        params = pq.WilsonParams(mass=0.0)

        result_plaq = pq.apply_M(U, psi, params, bc)
        result_quda = _apply_M_quda(U, psi, params, bc)

        print("\nFull gauge tracing test (simplified):")
        print("Gauge encoding: U_mu[0,0] = mu + 1 (1,2,3,4 for x,y,z,t)")
        print("Source site 0: parity=0, cb_idx=0")
        print()

        # For a source spinor at site 0, the hopping term contributes to neighbors
        # Forward hop to site x+mu: uses U_mu(site 0)
        # Backward hop from site 0: neighbor at x-mu receives contribution using U_mu(x-mu)^dag
        #
        # With mass=0, the Wilson operator is:
        # M*psi = -0.5 * sum_mu [ (1-gamma_mu)*U_mu(x)*psi(x+mu) + (1+gamma_mu)*U_mu^dag(x-mu)*psi(x-mu) ]
        #
        # The result at forward neighbor sites comes from the second term (backward hop in psi)
        # The result at backward neighbor sites comes from the first term (forward hop in psi)

        neighbor_sites = [
            (1, "x+ (result uses U_x(0)=1)", 0),
            (4, "y+ (result uses U_y(0)=2)", 1),
            (16, "z+ (result uses U_z(0)=3)", 2),
            (64, "t+ (result uses U_t(0)=4)", 3),
            (3, "x- (result uses U_x(3)=1)", 0),
            (12, "y- (result uses U_y(12)=2)", 1),
            (48, "z- (result uses U_z(48)=3)", 2),
            (192, "t- (result uses U_t(192)=4)", 3),
        ]

        print("Site | Direction | Expected U[0,0] | plaq s0c0 | quda s0c0")
        print("-" * 80)

        for site, desc, expected_mu in neighbor_sites:
            expected_gauge = expected_mu + 1
            val_plaq = result_plaq.site[site, 0, 0]
            val_quda = result_quda.site[site, 0, 0]
            diff = abs(val_plaq - val_quda)
            marker = "***" if diff > 1e-10 else ""
            print(
                f"{site:4d} | {desc:30s} | U={expected_gauge} | "
                f"{val_plaq.real:+.4f}{val_plaq.imag:+.4f}j | "
                f"{val_quda.real:+.4f}{val_quda.imag:+.4f}j {marker}"
            )

        diff = torch.abs(result_plaq.site - result_quda.site).max().item()
        print(f"\nMax diff: {diff}")

    def test_cb_idx_ordering_within_parity(self) -> None:
        """Investigate how QUDA orders sites within each parity class."""
        _lat = pq.Lattice((4, 4, 4, 4))

        print("\nCheckerboard index analysis:")
        print("Comparing plaq's cb_idx = site >> 1 with alternative orderings")
        print()

        # Check the first few sites of each parity
        print("Even sites (parity=0):")
        print("site | (x,y,z,t) | cb_idx=site>>1 | alt_cb (x-first) | alt2_cb (t-first)")
        print("-" * 80)

        even_count = 0
        for t in range(4):
            for z in range(4):
                for y in range(4):
                    for x in range(4):
                        if (x + y + z + t) % 2 == 0 and even_count < 16:
                            site = x + 4 * (y + 4 * (z + 4 * t))
                            cb_idx_shift = site >> 1

                            # Alternative: index = (x/2) + (Nx/2) * (y + Ny*(z + Nz*t))
                            # This matches what we do
                            alt_cb_x_first = (x // 2) + 2 * (y + 4 * (z + 4 * t))

                            # Alternative 2: t-first, then z, y, x
                            # index = (x + Nx*(y + Ny*z)) / 2 for fixed t
                            # This is more complex for 4D

                            print(
                                f"{site:4d} | ({x},{y},{z},{t}) | {cb_idx_shift:14d} | {alt_cb_x_first:16d}"
                            )
                            even_count += 1

        print()
        print("Odd sites (parity=1):")
        print("site | (x,y,z,t) | cb_idx=site>>1 | alt_cb (x-first)")
        print("-" * 80)

        odd_count = 0
        for t in range(4):
            for z in range(4):
                for y in range(4):
                    for x in range(4):
                        if (x + y + z + t) % 2 == 1 and odd_count < 16:
                            site = x + 4 * (y + 4 * (z + 4 * t))
                            cb_idx_shift = site >> 1
                            alt_cb_x_first = (x // 2) + 2 * (y + 4 * (z + 4 * t))

                            print(
                                f"{site:4d} | ({x},{y},{z},{t}) | {cb_idx_shift:14d} | {alt_cb_x_first:16d}"
                            )
                            odd_count += 1
