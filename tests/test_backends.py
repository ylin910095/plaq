"""Tests for backend abstraction layer."""

import pytest

from plaq.backends import Backend, BackendNotAvailableError, BackendRegistry, registry


class TestBackendRegistry:
    """Test backend registry functionality."""

    def test_plaq_backend_always_available(self) -> None:
        """Verify plaq backend is registered."""
        # Import plaq backend to trigger registration
        import plaq.backends.plaq  # noqa: F401

        assert registry.is_available(Backend.PLAQ)
        solver_fn = registry.get(Backend.PLAQ)
        assert callable(solver_fn)

    def test_unavailable_backend_raises(self) -> None:
        """Verify requesting unavailable backend raises BackendNotAvailableError."""
        # QUDA is not available (not registered)
        assert not registry.is_available(Backend.QUDA)

        with pytest.raises(BackendNotAvailableError) as exc_info:
            registry.get(Backend.QUDA)

        assert exc_info.value.backend == Backend.QUDA
        assert "QUDA" in str(exc_info.value)

    def test_registry_is_available(self) -> None:
        """Verify is_available() returns correct values."""
        # Import plaq backend to trigger registration
        import plaq.backends.plaq  # noqa: F401

        # Plaq should be available
        assert registry.is_available(Backend.PLAQ) is True

        # QUDA should not be available
        assert registry.is_available(Backend.QUDA) is False


class TestBackendRegistryUnit:
    """Unit tests for BackendRegistry class."""

    def test_register_and_get(self) -> None:
        """Test registering and retrieving a solver."""
        test_registry = BackendRegistry()

        def dummy_solver() -> str:
            return "dummy"

        test_registry.register(Backend.PLAQ, dummy_solver)
        assert test_registry.get(Backend.PLAQ) is dummy_solver

    def test_is_available_false_when_not_registered(self) -> None:
        """Test is_available returns False for unregistered backends."""
        test_registry = BackendRegistry()
        assert test_registry.is_available(Backend.PLAQ) is False
        assert test_registry.is_available(Backend.QUDA) is False

    def test_is_available_true_after_registration(self) -> None:
        """Test is_available returns True after registration."""
        test_registry = BackendRegistry()

        def dummy_solver() -> str:
            return "dummy"

        test_registry.register(Backend.PLAQ, dummy_solver)
        assert test_registry.is_available(Backend.PLAQ) is True

    def test_get_raises_for_unregistered(self) -> None:
        """Test get raises BackendNotAvailableError for unregistered backend."""
        test_registry = BackendRegistry()

        with pytest.raises(BackendNotAvailableError) as exc_info:
            test_registry.get(Backend.PLAQ)

        assert exc_info.value.backend == Backend.PLAQ


class TestBackendNotAvailableError:
    """Tests for BackendNotAvailableError exception."""

    def test_error_message(self) -> None:
        """Test error message contains backend name."""
        error = BackendNotAvailableError(Backend.QUDA)
        assert "QUDA" in str(error)
        assert error.backend == Backend.QUDA

    def test_error_is_exception(self) -> None:
        """Test BackendNotAvailableError is an Exception."""
        error = BackendNotAvailableError(Backend.PLAQ)
        assert isinstance(error, Exception)
