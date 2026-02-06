"""QUDA backend implementation for plaq solvers.

This package provides the QUDA library implementation of solvers and operators.
QUDA (https://github.com/lattice/quda) is a library for performing calculations
in lattice QCD on GPUs.

The QUDA backend requires the quda_torch_op package to be installed separately:

    uv pip install -e packages/quda_torch_op --no-build-isolation

The backend is automatically registered with the backend registry when this
module is imported (if quda_torch_op is available).

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plaq.backends import Backend, registry

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from plaq.fields import GaugeField, SpinorField
    from plaq.lattice import BoundaryCondition
    from plaq.operators import WilsonParams
    from plaq.solvers.api import SolverInfo

# Import quda_torch_op to verify it's available and trigger operator registration
import quda_torch_op

# Re-export for convenience
simple_add = quda_torch_op.simple_add


def quda_solve(
    U: GaugeField,
    b: SpinorField,
    action: str = "wilson",
    method: str = "auto",
    equation: str = "auto",
    tol: float = 1e-10,
    maxiter: int = 1000,
    precond: str | None = None,
    _dtype: torch.dtype | None = None,
    params: WilsonParams | None = None,
    bc: BoundaryCondition | None = None,
    callback: Callable[[SpinorField], None] | None = None,
) -> tuple[SpinorField, SolverInfo]:
    """Solve a lattice QCD linear system using the QUDA backend.

    Uses QUDA for operator applications (M, Mdag, MdagM) with plaq's
    Krylov solvers (CG/BiCGStab). QUDA operators are rescaled from
    QUDA convention (M_QUDA = 1 - kappa*D) to plaq convention
    (M_plaq = (m0+4) - D/2 = M_QUDA / (2*kappa)).

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
    equation : str
        Equation type: "M", "MdagM", or "auto".
    tol : float
        Relative tolerance for convergence.
    maxiter : int
        Maximum number of iterations.
    precond : str, optional
        Preconditioner type. Not supported in QUDA backend.
    _dtype : torch.dtype, optional
        Data type for computation (unused, kept for API compatibility).
    params : WilsonParams, optional
        Wilson operator parameters.
    bc : BoundaryCondition, optional
        Boundary conditions.
    callback : Callable[[SpinorField], None], optional
        User-supplied function to call after each iteration.
        Called as callback(xk) where xk is the current solution spinor field.

    Returns
    -------
    tuple[SpinorField, SolverInfo]
        Solution spinor field and convergence information.

    Raises
    ------
    ValueError
        If an unsupported action, method, equation, or preconditioner is specified.

    """
    from plaq.backends.plaq import bicgstab, cg
    from plaq.fields import SpinorField
    from plaq.lattice import BoundaryCondition
    from plaq.operators import WilsonParams
    from plaq.solvers.api import SolverInfo

    if action != "wilson":
        msg = f"Unsupported action: {action}. Only 'wilson' is supported."
        raise ValueError(msg)

    # Set defaults
    if params is None:
        params = WilsonParams(mass=0.1)
    if bc is None:
        bc = BoundaryCondition()

    # Auto-select equation type
    if equation == "auto":
        equation = "MdagM"

    if equation not in ("M", "MdagM"):
        msg = f"Unsupported equation type: {equation}. Use 'M' or 'MdagM'."
        raise ValueError(msg)

    # Auto-select solver method
    if method == "auto":
        method = "cg" if equation == "MdagM" else "bicgstab"

    if method not in ("cg", "bicgstab"):
        msg = f"Unsupported method: {method}. Use 'cg' or 'bicgstab'."
        raise ValueError(msg)

    # Reject preconditioning (not yet supported in QUDA backend)
    if precond is not None:
        msg = f"Preconditioner '{precond}' is not supported in the QUDA backend."
        raise ValueError(msg)

    # Auto-initialize QUDA if needed
    if not quda_torch_op.quda_is_initialized():
        quda_torch_op.quda_init(-1)

    # Extract data
    lattice = b.lattice
    gauge_data = U.data
    b_data = b.site
    kappa = params.kappa
    antiperiodic_t = bc.get_bc_phase(3) < 0

    # Rescaling factor: M_plaq = M_QUDA / (2*kappa)
    scale_M = 1.0 / (2.0 * kappa)
    scale_MdagM = scale_M * scale_M

    # Build operator application function
    if equation == "M":

        def A_apply(x_data: torch.Tensor) -> torch.Tensor:
            result = quda_torch_op.quda_wilson_mat(gauge_data, x_data, kappa, antiperiodic_t)
            return result * scale_M

        rhs = b_data
    else:
        # equation == "MdagM"

        def A_apply(x_data: torch.Tensor) -> torch.Tensor:
            result = quda_torch_op.quda_wilson_mat_dag_mat(
                gauge_data, x_data, kappa, antiperiodic_t
            )
            return result * scale_MdagM

        # Compute Mdag b in plaq convention
        rhs_quda = quda_torch_op.quda_wilson_mat_dag(gauge_data, b_data, kappa, antiperiodic_t)
        rhs = rhs_quda * scale_M

    # Wrap callback if provided
    tensor_callback = None
    if callback is not None:
        callback_fn = callback

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
        backend="quda",
    )

    return x, info


# Register QUDA backend on import
registry.register(Backend.QUDA, quda_solve)

__all__ = [
    "quda_solve",
    "simple_add",
]
