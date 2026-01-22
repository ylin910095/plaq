"""Backend abstraction layer for plaq solvers.

This module provides a backend registry that allows plaq to dispatch solver
calls to different implementations (native plaq vs QUDA).

Example
-------
>>> from plaq.backends import Backend, registry
>>> registry.is_available(Backend.PLAQ)
True
>>> solver_fn = registry.get(Backend.PLAQ)

"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class Backend(Enum):
    """Available backend implementations for solvers.

    Attributes
    ----------
    PLAQ : auto
        Native plaq implementation (always available).
    QUDA : auto
        QUDA library implementation (requires separate installation).

    """

    PLAQ = auto()
    QUDA = auto()


class BackendNotAvailableError(Exception):
    """Raised when a requested backend is not available.

    Parameters
    ----------
    backend : Backend
        The backend that was requested but not available.

    """

    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        super().__init__(f"Backend '{backend.name}' is not available.")


class BackendRegistry:
    """Registry for backend solver implementations.

    This class tracks available backends and provides methods to register
    and retrieve solver functions.

    Attributes
    ----------
    _solvers : dict[Backend, Callable]
        Mapping from backend to solver function.

    Example
    -------
    >>> registry = BackendRegistry()
    >>> registry.register(Backend.PLAQ, my_solver_fn)
    >>> registry.is_available(Backend.PLAQ)
    True
    >>> solver = registry.get(Backend.PLAQ)

    """

    def __init__(self) -> None:
        self._solvers: dict[Backend, Callable[..., Any]] = {}

    def register(self, backend: Backend, solver_fn: Callable[..., Any]) -> None:
        """Register a solver function for a backend.

        Parameters
        ----------
        backend : Backend
            The backend to register.
        solver_fn : Callable
            The solver function to associate with the backend.

        """
        self._solvers[backend] = solver_fn

    def get(self, backend: Backend) -> Callable[..., Any]:
        """Get the solver function for a backend.

        Parameters
        ----------
        backend : Backend
            The backend to retrieve.

        Returns
        -------
        Callable
            The solver function for the backend.

        Raises
        ------
        BackendNotAvailableError
            If the requested backend is not registered.

        """
        if backend not in self._solvers:
            raise BackendNotAvailableError(backend)
        return self._solvers[backend]

    def is_available(self, backend: Backend) -> bool:
        """Check if a backend is available.

        Parameters
        ----------
        backend : Backend
            The backend to check.

        Returns
        -------
        bool
            True if the backend is registered and available.

        """
        return backend in self._solvers


# Module-level registry instance
registry = BackendRegistry()

__all__ = [
    "Backend",
    "BackendNotAvailableError",
    "BackendRegistry",
    "registry",
]
