"""Tests for QUDA backend integration.

These tests verify the quda_torch_op package integration with plaq.
Tests are skipped if quda_torch_op is not installed.
"""

import pytest
import torch

# Check if quda_torch_op is available
try:
    import quda_torch_op  # noqa: F401 - needed to register ops

    HAS_QUDA_TORCH_OP = True
except ImportError:
    HAS_QUDA_TORCH_OP = False


pytestmark = pytest.mark.skipif(
    not HAS_QUDA_TORCH_OP,
    reason="quda_torch_op not installed",
)


class TestQudaTorchOpPackage:
    """Tests for the quda_torch_op package itself."""

    def test_import_quda_torch_op(self) -> None:
        """Verify quda_torch_op can be imported."""
        import quda_torch_op

        assert hasattr(quda_torch_op, "__version__")
        assert hasattr(quda_torch_op, "simple_add")

    def test_simple_add_basic(self) -> None:
        """Test basic simple_add functionality."""
        import quda_torch_op

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        result = quda_torch_op.simple_add(a, b)

        expected = torch.tensor([5.0, 7.0, 9.0])
        assert torch.allclose(result, expected)

    def test_simple_add_via_torch_ops(self) -> None:
        """Test simple_add via torch.ops namespace."""
        import quda_torch_op  # noqa: F401 - needed to register ops

        a = torch.randn(10, 20)
        b = torch.randn(10, 20)
        result = torch.ops.quda_torch_op.simple_add(a, b)

        assert torch.allclose(result, a + b)


class TestQudaBackendRegistration:
    """Tests for QUDA backend registration in plaq."""

    def test_quda_backend_available_after_import(self) -> None:
        """Verify QUDA backend is available after importing plaq.backends.quda."""
        # Import the QUDA backend
        from plaq.backends import (
            Backend,
            quda,  # noqa: F401
            registry,
        )

        assert registry.is_available(Backend.QUDA)

    def test_quda_backend_solver_registered(self) -> None:
        """Verify QUDA solver function is registered."""
        # Import the QUDA backend
        from plaq.backends import (
            Backend,
            quda,  # noqa: F401
            registry,
        )

        solver_fn = registry.get(Backend.QUDA)
        assert callable(solver_fn)

    def test_quda_simple_add_via_backend(self) -> None:
        """Test simple_add via plaq.backends.quda."""
        from plaq.backends import quda

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        result = quda.simple_add(a, b)

        expected = torch.tensor([5.0, 7.0, 9.0])
        assert torch.allclose(result, expected)


class TestQudaSolverBasic:
    """Tests for QUDA solver basic functionality."""

    def test_quda_solve_mdagm_identity_gauge(self) -> None:
        """Test quda_solve with equation='MdagM' on identity gauge (CG path)."""
        import plaq as pq
        from plaq.backends.quda import quda_solve

        lat = pq.Lattice((4, 4, 4, 8))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        b = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)

        x, info = quda_solve(U, b, equation="MdagM", params=params, bc=bc, tol=1e-10)

        assert info.converged
        assert info.backend == "quda"
        assert info.method == "cg"
        assert info.equation == "MdagM"

        # Verify solution: MdagM x = Mdag b
        from plaq.operators import apply_Mdag, apply_MdagM

        Mdag_b = apply_Mdag(U, b, params, bc)
        MdagM_x = apply_MdagM(U, x, params, bc)
        res_norm = torch.norm(MdagM_x.site - Mdag_b.site) / torch.norm(Mdag_b.site)
        assert res_norm < 1e-8

    def test_quda_solve_m_identity_gauge(self) -> None:
        """Test quda_solve with equation='M' on identity gauge (BiCGStab path)."""
        import plaq as pq
        from plaq.backends.quda import quda_solve

        lat = pq.Lattice((4, 4, 4, 8))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        b = pq.SpinorField.random(lat)
        params = pq.WilsonParams(mass=0.1)

        x, info = quda_solve(U, b, equation="M", params=params, bc=bc, tol=1e-10)

        assert info.converged
        assert info.backend == "quda"
        assert info.method == "bicgstab"
        assert info.equation == "M"

        # Verify M x = b
        from plaq.operators import apply_M

        Mx = apply_M(U, x, params, bc)
        res_norm = torch.norm(Mx.site - b.site) / torch.norm(b.site)
        assert res_norm < 1e-8
