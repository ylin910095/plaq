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
    dtype: torch.dtype | None = None,
    params: WilsonParams | None = None,
    bc: BoundaryCondition | None = None,
) -> tuple[SpinorField, SolverInfo]:
    """Solve a lattice QCD linear system using the QUDA backend.

    This is the QUDA backend implementation of the solver API.

    .. warning::

        QUDA solver operators are not yet implemented. This function currently
        raises NotImplementedError. The infrastructure is in place for future
        QUDA operator integration.

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
        Preconditioner type.
    dtype : torch.dtype, optional
        Data type for computation.
    params : WilsonParams, optional
        Wilson operator parameters.
    bc : BoundaryCondition, optional
        Boundary conditions.

    Returns
    -------
    tuple[SpinorField, SolverInfo]
        Solution spinor field and convergence information.

    Raises
    ------
    NotImplementedError
        QUDA solver operators are not yet implemented.

    """
    raise NotImplementedError(
        "QUDA solver operators are not yet implemented. "
        "The quda_torch_op package is installed and operators can be registered, "
        "but the actual QUDA Wilson/Dirac operators need to be added to csrc/code.cpp."
    )


# Register QUDA backend on import
registry.register(Backend.QUDA, quda_solve)

__all__ = [
    "quda_solve",
    "simple_add",
]
