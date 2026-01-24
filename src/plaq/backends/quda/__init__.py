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

import torch

from plaq.backends import Backend, registry
from plaq.fields import GaugeField, SpinorField
from plaq.lattice import BoundaryCondition
from plaq.operators import WilsonParams

if TYPE_CHECKING:
    from collections.abc import Callable

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
    dtype: torch.dtype | None = None,
    params: WilsonParams | None = None,
    bc: BoundaryCondition | None = None,
    callback: Callable[[SpinorField], None] | None = None,
) -> tuple[SpinorField, SolverInfo]:
    """Solve a lattice QCD linear system using the QUDA backend.

    This is the QUDA backend implementation of the solver API. It uses GPU-accelerated
    solvers from the QUDA library for high-performance inversions.

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
        Preconditioner type. Not currently supported for QUDA backend.
    dtype : torch.dtype, optional
        Data type for computation. Defaults to gauge field dtype.
    params : WilsonParams, optional
        Wilson operator parameters. If None, uses default mass=0.1.
    bc : BoundaryCondition, optional
        Boundary conditions. If None, uses default (antiperiodic in time).
    callback : Callable[[SpinorField], None], optional
        Not supported for QUDA backend (ignored).

    Returns
    -------
    tuple[SpinorField, SolverInfo]
        Solution spinor field and convergence information.

    Raises
    ------
    ValueError
        If an unsupported action or preconditioner is specified.
    RuntimeError
        If QUDA is not available or not initialized.

    Notes
    -----
    The QUDA backend automatically initializes QUDA on first use if not already
    initialized. The gauge field and spinor data are transferred to the GPU,
    the solve is performed, and results are transferred back to CPU.

    Storage format conversion is handled internally:
    - plaq gauge format [4, V, 3, 3] is converted to QUDA QDP format
    - plaq spinor format [V, 4, 3] is converted to QUDA even-odd format
    - Results are converted back to plaq format

    """
    from plaq.solvers.api import SolverInfo

    if action != "wilson":
        msg = f"Unsupported action: {action}. Only 'wilson' is supported."
        raise ValueError(msg)

    if precond is not None:
        msg = f"Preconditioner '{precond}' not supported for QUDA backend."
        raise ValueError(msg)

    # Check QUDA availability
    if not quda_torch_op.quda_is_available():
        msg = (
            "QUDA is not available. Ensure quda_torch_op was built with QUDA support. "
            "Set QUDA_HOME and MPI_HOME environment variables and rebuild."
        )
        raise RuntimeError(msg)

    # Auto-initialize QUDA if needed
    if not quda_torch_op.quda_is_initialized():
        quda_torch_op.quda_init(-1)  # Use default GPU

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

    # Auto-select solver method (for info purposes - QUDA handles internally)
    if method == "auto":
        method = "cg" if equation == "MdagM" else "bicgstab"

    # Get lattice information
    lattice = b.lattice
    dims = lattice.shape

    # Determine working dtype
    if dtype is None:
        dtype = U.dtype

    # Convert gauge field to contiguous tensor with correct dtype
    gauge_data = U.data  # [4, V, 3, 3]
    if gauge_data.dtype != dtype:
        gauge_data = gauge_data.to(dtype=dtype)
    gauge_data = gauge_data.contiguous()

    # Convert spinor to site layout with correct dtype
    spinor_data = b.site  # [V, 4, 3]
    if spinor_data.dtype != dtype:
        spinor_data = spinor_data.to(dtype=dtype)
    spinor_data = spinor_data.contiguous()

    # Prepare lattice dimensions tensor
    dims_tensor = torch.tensor([dims[0], dims[1], dims[2], dims[3]], dtype=torch.int64)

    # Compute kappa from Wilson params
    kappa = params.kappa

    # Determine temporal boundary condition
    t_boundary = int(bc.fermion_bc_time)  # -1 for antiperiodic, +1 for periodic

    # Ensure data is on CPU (QUDA expects CPU tensors, handles GPU internally)
    if gauge_data.device.type != "cpu":
        gauge_data = gauge_data.cpu()
    if spinor_data.device.type != "cpu":
        spinor_data = spinor_data.cpu()

    # Call QUDA solver
    solution_data, converged, iters, residual = quda_torch_op.wilson_invert(
        gauge_data,
        spinor_data,
        dims_tensor,
        kappa,
        tol,
        maxiter,
        equation,
        t_boundary,
    )

    # Wrap result in SpinorField
    x = SpinorField(solution_data, lattice, layout="site")

    # Create solver info
    info = SolverInfo(
        converged=converged,
        iters=iters,
        final_residual=residual,
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
