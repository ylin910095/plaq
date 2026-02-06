"""High-level solver API for lattice QCD linear systems.

This module provides the main :func:`solve` function that automatically
selects the appropriate solver and equation type based on the action.

The API supports:

- **equation="M"**: Solve :math:`M x = b` directly using BiCGStab
- **equation="MdagM"**: Solve :math:`M^\\dagger M x = M^\\dagger b` using CG

For the Wilson action, the default is "MdagM" which gives a Hermitian
positive-definite system suitable for CG.

The actual solver implementation is provided through the backend abstraction
layer (:mod:`plaq.backends`). The native plaq backend (:mod:`plaq.backends.plaq`)
contains the implementation and is always available.

Example
-------
>>> import plaq as pq
>>> lat = pq.Lattice((4, 4, 4, 8))
>>> bc = pq.BoundaryCondition()
>>> lat.build_neighbor_tables(bc)
>>> U = pq.GaugeField.eye(lat)
>>> b = pq.SpinorField.random(lat)
>>> params = pq.WilsonParams(mass=0.1)
>>> x, info = pq.solve(U, b, params=params, bc=bc)
>>> print(f"Converged: {info.converged}, iters: {info.iters}")

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from plaq.backends.plaq.bicgstab import bicgstab
from plaq.backends.plaq.cg import cg
from plaq.fields import GaugeField, SpinorField
from plaq.lattice import BoundaryCondition
from plaq.operators import WilsonParams, apply_M, apply_Mdag, apply_MdagM

if TYPE_CHECKING:
    from collections.abc import Callable

    from plaq.lattice import Lattice


@dataclass
class SolverInfo:
    """Information about solver convergence.

    Attributes
    ----------
    converged : bool
        Whether the solver converged within tolerance.
    iters : int
        Number of iterations performed.
    final_residual : float
        Final relative residual norm.
    method : str
        Solver method used ("cg" or "bicgstab").
    equation : str
        Equation type solved ("M" or "MdagM").
    backend : str
        Backend used for the solve ("plaq" or "quda").

    """

    converged: bool
    iters: int
    final_residual: float
    method: str
    equation: str
    backend: str


def solve(
    U: GaugeField,
    b: SpinorField,
    action: str = "wilson",
    method: str = "auto",
    equation: str = "auto",
    tol: float = 1e-10,
    maxiter: int = 1000,
    precond: str | None = None,
    dtype: torch.dtype = torch.complex128,
    params: WilsonParams | None = None,
    bc: BoundaryCondition | None = None,
    callback: Callable[[SpinorField], None] | None = None,
    backend: str = "auto",
) -> tuple[SpinorField, SolverInfo]:
    """Solve a lattice QCD linear system.

    High-level API for solving linear systems involving the Dirac operator.

    For **equation="M"** (direct solve):

    .. math::

        M x = b

    Solved using BiCGStab for the non-Hermitian Wilson operator.

    For **equation="MdagM"** (normal equation):

    .. math::

        M^\\dagger M x = M^\\dagger b

    Solved using CG for the Hermitian positive-definite normal operator.

    Parameters
    ----------
    U : GaugeField
        Gauge field configuration.
    b : SpinorField
        Right-hand side (source) spinor field.
    action : str
        Action type. Currently only "wilson" is supported.
    method : str
        Solver method: "cg", "bicgstab", or "auto".
        If "auto", selects CG for MdagM and BiCGStab for M.
    equation : str
        Equation type: "M", "MdagM", or "auto".
        If "auto", defaults to "MdagM" for Wilson action.
    tol : float
        Relative tolerance for convergence.
    maxiter : int
        Maximum number of iterations.
    precond : str, optional
        Preconditioner type. Currently "eo" (even-odd) is supported for MdagM.
    dtype : torch.dtype
        Data type for computation. Default is complex128.
    params : WilsonParams, optional
        Wilson operator parameters. If None, uses default mass=0.1.
    bc : BoundaryCondition, optional
        Boundary conditions. If None, uses default (antiperiodic in time).
    callback : Callable[[SpinorField], None], optional
        User-supplied function to call after each iteration.
        Called as callback(xk) where xk is the current solution spinor field.
    backend : str
        Backend to use: "auto", "plaq", or "quda".
        If "auto", uses QUDA if available and tensors are on CUDA, otherwise plaq.

    Returns
    -------
    tuple[SpinorField, SolverInfo]
        Solution spinor field and convergence information.

    Raises
    ------
    ValueError
        If an unsupported action, method, equation, or preconditioner is specified.
    BackendNotAvailableError
        If the requested backend is not available.

    Example
    -------
    >>> import plaq as pq
    >>> lat = pq.Lattice((4, 4, 4, 8))
    >>> bc = pq.BoundaryCondition()
    >>> lat.build_neighbor_tables(bc)
    >>> U = pq.GaugeField.eye(lat)
    >>> b = pq.SpinorField.random(lat)
    >>> x, info = pq.solve(U, b)
    >>> print(f"Converged: {info.converged}, backend: {info.backend}")

    """
    # Import plaq backend to ensure it's registered
    import plaq.backends.plaq  # noqa: F401
    from plaq.backends import Backend, BackendNotAvailableError, registry

    def _try_import_quda_backend() -> bool:
        """Try to import the QUDA backend, returning True if successful."""
        try:
            import plaq.backends.quda  # noqa: F401

            return True
        except ImportError:
            return False

    if action != "wilson":
        msg = f"Unsupported action: {action}. Only 'wilson' is supported."
        raise ValueError(msg)

    # Validate and select backend
    if backend not in ("auto", "plaq", "quda"):
        msg = f"Unsupported backend: {backend}. Use 'auto', 'plaq', or 'quda'."
        raise ValueError(msg)

    # Determine which backend to use
    if backend == "auto":
        # Try to import QUDA backend for auto-detection
        _try_import_quda_backend()
        # Use QUDA if available and tensors are on CUDA
        is_cuda = b.site.is_cuda
        backend_name = "quda" if is_cuda and registry.is_available(Backend.QUDA) else "plaq"
    elif backend == "plaq":
        backend_name = "plaq"
    else:  # backend == "quda"
        # Try to import QUDA backend first
        _try_import_quda_backend()
        if not registry.is_available(Backend.QUDA):
            raise BackendNotAvailableError(Backend.QUDA)
        # QUDA backend is available, dispatch to it
        from plaq.backends.quda import quda_solve

        return quda_solve(
            U, b, action, method, equation, tol, maxiter, precond, dtype, params, bc, callback
        )

    # Set defaults
    if params is None:
        params = WilsonParams(mass=0.1)
    if bc is None:
        bc = BoundaryCondition()

    # Auto-select equation type
    if equation == "auto":
        equation = "MdagM"  # Default for Wilson

    if equation not in ("M", "MdagM"):
        msg = f"Unsupported equation type: {equation}. Use 'M' or 'MdagM'."
        raise ValueError(msg)

    # Auto-select solver method
    if method == "auto":
        method = "cg" if equation == "MdagM" else "bicgstab"

    if method not in ("cg", "bicgstab"):
        msg = f"Unsupported method: {method}. Use 'cg' or 'bicgstab'."
        raise ValueError(msg)

    # Ensure neighbor tables are built
    lattice = b.lattice
    try:
        lattice.neighbor_fwd(0)
    except RuntimeError:
        lattice.build_neighbor_tables(bc)

    # Convert to working dtype
    b_data = b.site.to(dtype=dtype)

    # Convert gauge field to working dtype if needed
    U_data = U.data
    if U_data.dtype != dtype:
        U_data = U_data.to(dtype=dtype)
        U_work = GaugeField(U_data, lattice)
    else:
        U_work = U

    # Handle preconditioning
    if precond == "eo":
        return _solve_preconditioned_eo(
            U_work,
            b_data,
            lattice,
            params,
            bc,
            equation,
            method,
            tol,
            maxiter,
            dtype,
            callback,
            backend_name,
        )
    elif precond is not None:
        msg = f"Unsupported preconditioner: {precond}. Use 'eo' or None."
        raise ValueError(msg)

    # Build operator application function
    if equation == "M":
        # Solve M x = b

        def A_apply(x_data: torch.Tensor) -> torch.Tensor:
            x_field = SpinorField(x_data, lattice, layout="site")
            result = apply_M(U_work, x_field, params, bc)
            return result.site

        rhs = b_data
    else:
        # Solve MdagM x = Mdag b

        def A_apply(x_data: torch.Tensor) -> torch.Tensor:
            x_field = SpinorField(x_data, lattice, layout="site")
            result = apply_MdagM(U_work, x_field, params, bc)
            return result.site

        # Compute Mdag b (need to work with converted dtype spinor)
        b_field_work = SpinorField(b_data, lattice, layout="site")
        rhs_field = apply_Mdag(U_work, b_field_work, params, bc)
        rhs = rhs_field.site

    # Wrap callback if provided
    tensor_callback = None
    if callback is not None:
        callback_fn = callback  # Capture in local scope for type checker

        def tensor_callback(x_data: torch.Tensor) -> None:
            x_field = SpinorField(x_data, lattice, layout="site")
            callback_fn(x_field)

    # Select and run solver
    if method == "cg":
        x_data, solver_info = cg(A_apply, rhs, tol=tol, maxiter=maxiter, callback=tensor_callback)
    else:
        x_data, solver_info = bicgstab(
            A_apply, rhs, tol=tol, maxiter=maxiter, callback=tensor_callback
        )

    # Wrap result
    x = SpinorField(x_data, lattice, layout="site")
    info = SolverInfo(
        converged=solver_info.converged,
        iters=solver_info.iters,
        final_residual=solver_info.final_residual,
        method=method,
        equation=equation,
        backend=backend_name,
    )

    return x, info


def _solve_preconditioned_eo(
    U: GaugeField,
    b_data: torch.Tensor,
    lattice: Lattice,
    params: WilsonParams,
    bc: BoundaryCondition,
    equation: str,
    _method: str,
    tol: float,
    maxiter: int,
    dtype: torch.dtype,
    callback: Callable[[SpinorField], None] | None = None,
    backend_name: str = "plaq",
) -> tuple[SpinorField, SolverInfo]:
    """Solve with even-odd preconditioning.

    This function implements Schur complement preconditioning for MdagM.

    """
    from plaq.precond.even_odd import solve_eo_preconditioned

    if equation != "MdagM":
        msg = "Even-odd preconditioning is only supported for equation='MdagM'."
        raise ValueError(msg)

    # Wrap callback if provided (for EO solver which works with site layout)
    tensor_callback = None
    if callback is not None:
        callback_fn = callback  # Capture in local scope for type checker

        def tensor_callback(x_data: torch.Tensor) -> None:
            x_field = SpinorField(x_data, lattice, layout="site")
            callback_fn(x_field)

    # Delegate to EO solver
    x_data, solver_info = solve_eo_preconditioned(
        U, b_data, lattice, params, bc, tol, maxiter, dtype, tensor_callback
    )

    x = SpinorField(x_data, lattice, layout="site")
    info = SolverInfo(
        converged=solver_info.converged,
        iters=solver_info.iters,
        final_residual=solver_info.final_residual,
        method="cg",
        equation="MdagM",
        backend=backend_name,
    )

    return x, info
