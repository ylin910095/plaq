"""Tests for solver backend selection."""

import pytest

import plaq as pq


class TestSolverBackendSelection:
    """Test backend selection in solve() function."""

    def test_solve_default_backend_is_plaq_on_cpu(self) -> None:
        """Verify CPU tensors use plaq backend by default."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        b = pq.SpinorField.random(lat)

        # Default backend should be plaq on CPU
        _x, info = pq.solve(U, b, tol=1e-8)

        assert info.converged
        assert info.backend == "plaq"

    def test_solve_explicit_plaq_backend(self) -> None:
        """Verify backend='plaq' works explicitly."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        b = pq.SpinorField.random(lat)

        # Explicit plaq backend
        _x, info = pq.solve(U, b, backend="plaq", tol=1e-8)

        assert info.converged
        assert info.backend == "plaq"

    def test_solve_quda_backend(self) -> None:
        """Verify backend='quda' behavior.

        When quda_torch_op is not installed, raises BackendNotAvailableError.
        When quda_torch_op is installed, solves successfully.
        """
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        b = pq.SpinorField.random(lat)

        # Check if quda_torch_op is installed
        try:
            import quda_torch_op  # noqa: F401

            has_quda = True
        except ImportError:
            has_quda = False

        if has_quda:
            # QUDA package is installed, solve should succeed
            _x, info = pq.solve(U, b, backend="quda", tol=1e-8)
            assert info.converged
            assert info.backend == "quda"
        else:
            # QUDA package is not installed
            with pytest.raises(pq.BackendNotAvailableError) as exc_info:
                pq.solve(U, b, backend="quda")

            assert "QUDA" in str(exc_info.value)

    def test_solver_info_includes_backend(self) -> None:
        """Verify SolverInfo.backend is populated correctly."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        b = pq.SpinorField.random(lat)

        # Test with default backend
        _x, info = pq.solve(U, b, tol=1e-8)
        assert hasattr(info, "backend")
        assert isinstance(info.backend, str)
        assert info.backend == "plaq"

        # Test with explicit backend
        _x, info = pq.solve(U, b, backend="plaq", tol=1e-8)
        assert info.backend == "plaq"

    def test_solve_invalid_backend_raises(self) -> None:
        """Verify invalid backend raises ValueError."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        b = pq.SpinorField.random(lat)

        with pytest.raises(ValueError) as exc_info:
            pq.solve(U, b, backend="invalid")

        assert "Unsupported backend" in str(exc_info.value)

    def test_solve_auto_backend_on_cpu(self) -> None:
        """Verify backend='auto' selects plaq on CPU."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        b = pq.SpinorField.random(lat)

        # Auto backend should select plaq on CPU
        _x, info = pq.solve(U, b, backend="auto", tol=1e-8)

        assert info.converged
        assert info.backend == "plaq"

    def test_solve_with_preconditioning_includes_backend(self) -> None:
        """Verify backend is included when using preconditioning."""
        lat = pq.Lattice((4, 4, 4, 4))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        b = pq.SpinorField.random(lat)

        # Solve with even-odd preconditioning
        _x, info = pq.solve(U, b, precond="eo", tol=1e-8)

        assert info.converged
        assert info.backend == "plaq"


class TestBackendNotAvailableError:
    """Test BackendNotAvailableError export from plaq package."""

    def test_backend_not_available_error_exported(self) -> None:
        """Verify BackendNotAvailableError is exported from plaq."""
        assert hasattr(pq, "BackendNotAvailableError")
        assert pq.BackendNotAvailableError is not None

    def test_backend_not_available_error_is_exception(self) -> None:
        """Verify BackendNotAvailableError is an Exception."""
        from plaq.backends import Backend

        error = pq.BackendNotAvailableError(Backend.QUDA)
        assert isinstance(error, Exception)
        assert "QUDA" in str(error)
