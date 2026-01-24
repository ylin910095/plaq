"""Tests for QUDA Wilson solver comparing with CPU reference."""

import pytest
import torch

import plaq as pq


def has_quda() -> bool:
    """Check if QUDA backend is available."""
    try:
        import quda_torch_op

        return quda_torch_op.quda_is_available()
    except ImportError:
        return False


@pytest.fixture(scope="module")
def quda_initialized():
    """Initialize QUDA once for the module."""
    if not has_quda():
        pytest.skip("QUDA not available")

    import quda_torch_op

    if not quda_torch_op.quda_is_initialized():
        quda_torch_op.quda_init(0)
    yield
    # Don't finalize QUDA here as other tests might need it


@pytest.mark.skipif(not has_quda(), reason="QUDA not available")
class TestQudaWilsonSolver:
    """Test suite for QUDA Wilson solver."""

    def test_quda_wilson_invert_identity_gauge_MdagM(self, quda_initialized):
        """Test QUDA solve with identity gauge field (MdagM equation)."""
        # Create a small lattice
        lat = pq.Lattice((4, 4, 4, 8))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        # Identity gauge field
        U = pq.GaugeField.eye(lat, dtype=torch.complex128)

        # Random source
        torch.manual_seed(42)
        b = pq.SpinorField.random(lat, dtype=torch.complex128)

        # Wilson parameters
        params = pq.WilsonParams(mass=0.1)

        # Solve with QUDA
        x_quda, info_quda = pq.solve(
            U, b, backend="quda", params=params, bc=bc, equation="MdagM", tol=1e-10
        )

        # Solve with plaq (CPU reference)
        x_plaq, info_plaq = pq.solve(
            U, b, backend="plaq", params=params, bc=bc, equation="MdagM", tol=1e-10
        )

        print(f"\nQUDA: converged={info_quda.converged}, iters={info_quda.iters}, "
              f"residual={info_quda.final_residual:.2e}")
        print(f"plaq: converged={info_plaq.converged}, iters={info_plaq.iters}, "
              f"residual={info_plaq.final_residual:.2e}")

        # Compare solutions
        diff = torch.norm(x_quda.site - x_plaq.site) / torch.norm(x_plaq.site)
        print(f"Relative difference: {diff:.2e}")

        # Both should converge
        assert info_quda.converged, f"QUDA did not converge: {info_quda.final_residual}"
        assert info_plaq.converged, f"plaq did not converge: {info_plaq.final_residual}"

        # Solutions should match to reasonable precision
        assert diff < 1e-6, f"Solutions differ too much: {diff}"

    def test_quda_wilson_invert_random_gauge_MdagM(self, quda_initialized):
        """Test QUDA solve with random gauge field (MdagM equation)."""
        # Create a small lattice
        lat = pq.Lattice((4, 4, 4, 8))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        # Random SU(3) gauge field
        torch.manual_seed(123)
        U = pq.GaugeField.random(lat, dtype=torch.complex128)

        # Random source
        torch.manual_seed(456)
        b = pq.SpinorField.random(lat, dtype=torch.complex128)

        # Wilson parameters (heavier mass for faster convergence)
        params = pq.WilsonParams(mass=0.5)

        # Solve with QUDA
        x_quda, info_quda = pq.solve(
            U, b, backend="quda", params=params, bc=bc, equation="MdagM", tol=1e-10
        )

        # Solve with plaq (CPU reference)
        x_plaq, info_plaq = pq.solve(
            U, b, backend="plaq", params=params, bc=bc, equation="MdagM", tol=1e-10
        )

        print(f"\nQUDA: converged={info_quda.converged}, iters={info_quda.iters}, "
              f"residual={info_quda.final_residual:.2e}")
        print(f"plaq: converged={info_plaq.converged}, iters={info_plaq.iters}, "
              f"residual={info_plaq.final_residual:.2e}")

        # Compare solutions
        diff = torch.norm(x_quda.site - x_plaq.site) / torch.norm(x_plaq.site)
        print(f"Relative difference: {diff:.2e}")

        # Both should converge
        assert info_quda.converged, f"QUDA did not converge: {info_quda.final_residual}"
        assert info_plaq.converged, f"plaq did not converge: {info_plaq.final_residual}"

        # Solutions should match to reasonable precision
        assert diff < 1e-6, f"Solutions differ too much: {diff}"

    def test_quda_solution_satisfies_equation_MdagM(self, quda_initialized):
        """Verify QUDA solution satisfies M^dag M x = M^dag b."""
        lat = pq.Lattice((4, 4, 4, 8))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        torch.manual_seed(789)
        U = pq.GaugeField.random(lat, dtype=torch.complex128)
        b = pq.SpinorField.random(lat, dtype=torch.complex128)

        params = pq.WilsonParams(mass=0.3)

        # Solve with QUDA
        x, info = pq.solve(
            U, b, backend="quda", params=params, bc=bc, equation="MdagM", tol=1e-10
        )

        # Compute M^dag M x
        Mx = pq.apply_M(U, x, params, bc)
        MdagMx = pq.apply_Mdag(U, Mx, params, bc)

        # Compute M^dag b
        Mdagb = pq.apply_Mdag(U, b, params, bc)

        # Check residual: || M^dag M x - M^dag b || / || M^dag b ||
        residual = torch.norm(MdagMx.site - Mdagb.site) / torch.norm(Mdagb.site)
        print(f"\nTrue residual: {residual:.2e}")

        assert residual < 1e-8, f"Solution does not satisfy equation: {residual}"

    def test_quda_wilson_invert_M_equation(self, quda_initialized):
        """Test QUDA solve with direct M equation (BiCGStab)."""
        lat = pq.Lattice((4, 4, 4, 8))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        torch.manual_seed(321)
        U = pq.GaugeField.random(lat, dtype=torch.complex128)
        b = pq.SpinorField.random(lat, dtype=torch.complex128)

        params = pq.WilsonParams(mass=0.5)

        # Solve M x = b with QUDA
        x_quda, info_quda = pq.solve(
            U, b, backend="quda", params=params, bc=bc, equation="M", tol=1e-10
        )

        # Solve with plaq (CPU reference)
        x_plaq, info_plaq = pq.solve(
            U, b, backend="plaq", params=params, bc=bc, equation="M", tol=1e-10
        )

        print(f"\nQUDA: converged={info_quda.converged}, iters={info_quda.iters}, "
              f"residual={info_quda.final_residual:.2e}")
        print(f"plaq: converged={info_plaq.converged}, iters={info_plaq.iters}, "
              f"residual={info_plaq.final_residual:.2e}")

        # Verify QUDA solution satisfies M x = b
        Mx = pq.apply_M(U, x_quda, params, bc)
        residual = torch.norm(Mx.site - b.site) / torch.norm(b.site)
        print(f"True residual: {residual:.2e}")

        assert info_quda.converged, f"QUDA did not converge"
        assert residual < 1e-8, f"Solution does not satisfy M x = b: {residual}"

    def test_quda_single_precision(self, quda_initialized):
        """Test QUDA solve with single precision."""
        lat = pq.Lattice((4, 4, 4, 8))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        torch.manual_seed(111)
        U = pq.GaugeField.random(lat, dtype=torch.complex64)
        b = pq.SpinorField.random(lat, dtype=torch.complex64)

        params = pq.WilsonParams(mass=0.5)

        # Solve with QUDA in single precision
        x, info = pq.solve(
            U, b, backend="quda", params=params, bc=bc, equation="MdagM", tol=1e-6
        )

        print(f"\nSingle precision: converged={info.converged}, "
              f"iters={info.iters}, residual={info.final_residual:.2e}")

        assert info.converged, f"QUDA single precision did not converge"
        assert x.dtype == torch.complex64, "Output dtype should be complex64"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
