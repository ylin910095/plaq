"""Comprehensive tests for QUDA solver backend.

Tests cross-backend comparison, mass sweeps, residual checks,
callback invocation, and SolverInfo correctness.
Skipped if quda_torch_op is not installed.
"""

import pytest
import torch

import plaq as pq
from plaq.operators import apply_M, apply_Mdag, apply_MdagM

# Check if quda_torch_op is available
try:
    import quda_torch_op  # noqa: F401

    HAS_QUDA = True
except ImportError:
    HAS_QUDA = False

pytestmark = pytest.mark.skipif(not HAS_QUDA, reason="quda_torch_op not installed")


@pytest.fixture()
def lattice_4444() -> pq.Lattice:
    """Create a 4^4 lattice with neighbor tables."""
    lat = pq.Lattice((4, 4, 4, 4))
    bc = pq.BoundaryCondition()
    lat.build_neighbor_tables(bc)
    return lat


@pytest.fixture()
def lattice_4448() -> pq.Lattice:
    """Create a 4^3 x 8 lattice with neighbor tables."""
    lat = pq.Lattice((4, 4, 4, 8))
    bc = pq.BoundaryCondition()
    lat.build_neighbor_tables(bc)
    return lat


class TestCrossBackendComparison:
    """Verify QUDA and plaq backends produce matching solutions."""

    def test_mdagm_identity_gauge(self, lattice_4448: pq.Lattice) -> None:
        """MdagM solve on identity gauge should match between backends."""
        lat = lattice_4448
        U = pq.GaugeField.eye(lat)
        torch.manual_seed(42)
        b = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)
        bc = pq.BoundaryCondition()

        x_plaq, info_plaq = pq.solve(
            U, b, equation="MdagM", params=params, bc=bc, backend="plaq", tol=1e-12
        )
        x_quda, info_quda = pq.solve(
            U, b, equation="MdagM", params=params, bc=bc, backend="quda", tol=1e-12
        )

        assert info_plaq.converged
        assert info_quda.converged
        assert info_plaq.backend == "plaq"
        assert info_quda.backend == "quda"

        diff = torch.norm(x_plaq.site - x_quda.site) / torch.norm(x_plaq.site)
        assert diff < 1e-8

    def test_m_identity_gauge(self, lattice_4448: pq.Lattice) -> None:
        """M solve on identity gauge should match between backends."""
        lat = lattice_4448
        U = pq.GaugeField.eye(lat)
        torch.manual_seed(123)
        b = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)
        bc = pq.BoundaryCondition()

        x_plaq, info_plaq = pq.solve(
            U, b, equation="M", params=params, bc=bc, backend="plaq", tol=1e-12
        )
        x_quda, info_quda = pq.solve(
            U, b, equation="M", params=params, bc=bc, backend="quda", tol=1e-12
        )

        assert info_plaq.converged
        assert info_quda.converged

        diff = torch.norm(x_plaq.site - x_quda.site) / torch.norm(x_plaq.site)
        assert diff < 1e-8

    def test_mdagm_random_gauge(self, lattice_4444: pq.Lattice) -> None:
        """MdagM solve on random gauge should match between backends."""
        lat = lattice_4444
        torch.manual_seed(99)
        U = pq.GaugeField.random(lat)
        b = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.5)
        bc = pq.BoundaryCondition()

        x_plaq, info_plaq = pq.solve(
            U, b, equation="MdagM", params=params, bc=bc, backend="plaq", tol=1e-12
        )
        x_quda, info_quda = pq.solve(
            U, b, equation="MdagM", params=params, bc=bc, backend="quda", tol=1e-12
        )

        assert info_plaq.converged
        assert info_quda.converged

        diff = torch.norm(x_plaq.site - x_quda.site) / torch.norm(x_plaq.site)
        assert diff < 1e-6

    def test_m_random_gauge(self, lattice_4444: pq.Lattice) -> None:
        """M solve on random gauge should match between backends."""
        lat = lattice_4444
        torch.manual_seed(77)
        U = pq.GaugeField.random(lat)
        b = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.5)
        bc = pq.BoundaryCondition()

        x_plaq, info_plaq = pq.solve(
            U, b, equation="M", params=params, bc=bc, backend="plaq", tol=1e-12
        )
        x_quda, info_quda = pq.solve(
            U, b, equation="M", params=params, bc=bc, backend="quda", tol=1e-12
        )

        assert info_plaq.converged
        assert info_quda.converged

        diff = torch.norm(x_plaq.site - x_quda.site) / torch.norm(x_plaq.site)
        assert diff < 1e-6


class TestMassParameterSweep:
    """Verify solver works across different mass values."""

    @pytest.mark.parametrize("mass", [0.1, 0.5, 1.0])
    def test_mdagm_mass_sweep(self, lattice_4444: pq.Lattice, mass: float) -> None:
        """MdagM solve should converge for various mass values."""
        lat = lattice_4444
        U = pq.GaugeField.eye(lat)
        torch.manual_seed(42)
        b = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=mass)
        bc = pq.BoundaryCondition()

        x, info = pq.solve(U, b, equation="MdagM", params=params, bc=bc, backend="quda", tol=1e-10)

        assert info.converged
        assert info.backend == "quda"

        # Verify residual with plaq operators
        Mdag_b = apply_Mdag(U, b, params, bc)
        MdagM_x = apply_MdagM(U, x, params, bc)
        res_norm = torch.norm(MdagM_x.site - Mdag_b.site) / torch.norm(Mdag_b.site)
        assert res_norm < 1e-8


class TestResidualCheck:
    """Self-contained residual verification using plaq operators."""

    def test_m_residual(self, lattice_4448: pq.Lattice) -> None:
        """Verify ||M x - b|| / ||b|| is small after solving M x = b."""
        lat = lattice_4448
        U = pq.GaugeField.eye(lat)
        torch.manual_seed(7)
        b = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)
        bc = pq.BoundaryCondition()

        x, info = pq.solve(U, b, equation="M", params=params, bc=bc, backend="quda", tol=1e-10)

        assert info.converged

        Mx = apply_M(U, x, params, bc)
        res_norm = torch.norm(Mx.site - b.site) / torch.norm(b.site)
        assert res_norm < 1e-8

    def test_mdagm_residual(self, lattice_4448: pq.Lattice) -> None:
        """Verify ||MdagM x - Mdag b|| / ||Mdag b|| is small."""
        lat = lattice_4448
        U = pq.GaugeField.eye(lat)
        torch.manual_seed(13)
        b = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)
        bc = pq.BoundaryCondition()

        x, info = pq.solve(U, b, equation="MdagM", params=params, bc=bc, backend="quda", tol=1e-10)

        assert info.converged

        Mdag_b = apply_Mdag(U, b, params, bc)
        MdagM_x = apply_MdagM(U, x, params, bc)
        res_norm = torch.norm(MdagM_x.site - Mdag_b.site) / torch.norm(Mdag_b.site)
        assert res_norm < 1e-8


class TestCallback:
    """Verify callback is invoked during QUDA solve."""

    def test_callback_called(self, lattice_4444: pq.Lattice) -> None:
        """Verify callback is called at least once during solve."""
        lat = lattice_4444
        U = pq.GaugeField.eye(lat)
        torch.manual_seed(42)
        b = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)
        bc = pq.BoundaryCondition()

        call_count = [0]

        def my_callback(xk: pq.SpinorField) -> None:
            call_count[0] += 1
            assert isinstance(xk, pq.SpinorField)

        _x, info = pq.solve(
            U,
            b,
            equation="MdagM",
            params=params,
            bc=bc,
            backend="quda",
            tol=1e-10,
            callback=my_callback,
        )

        assert info.converged
        assert call_count[0] > 0


class TestSolverInfoCorrectness:
    """Verify SolverInfo fields are populated correctly."""

    def test_info_mdagm(self, lattice_4444: pq.Lattice) -> None:
        """SolverInfo for MdagM should have correct fields."""
        lat = lattice_4444
        U = pq.GaugeField.eye(lat)
        torch.manual_seed(42)
        b = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)
        bc = pq.BoundaryCondition()

        _x, info = pq.solve(U, b, equation="MdagM", params=params, bc=bc, backend="quda", tol=1e-10)

        assert info.backend == "quda"
        assert info.method == "cg"
        assert info.equation == "MdagM"
        assert info.converged is True
        assert info.iters > 0
        assert info.final_residual < 1e-10

    def test_info_m(self, lattice_4444: pq.Lattice) -> None:
        """SolverInfo for M should have correct fields."""
        lat = lattice_4444
        U = pq.GaugeField.eye(lat)
        torch.manual_seed(42)
        b = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)
        bc = pq.BoundaryCondition()

        _x, info = pq.solve(U, b, equation="M", params=params, bc=bc, backend="quda", tol=1e-10)

        assert info.backend == "quda"
        assert info.method == "bicgstab"
        assert info.equation == "M"
        assert info.converged is True
        assert info.iters > 0
        assert info.final_residual < 1e-10
