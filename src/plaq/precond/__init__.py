"""Preconditioning subpackage for plaq.

This subpackage provides preconditioning methods for lattice QCD solvers:

- **Even-odd (Schur complement)**: Reduces the system size by a factor of 2
  by exploiting the checkerboard structure of the Wilson operator.

"""

from plaq.precond.even_odd import solve_eo_preconditioned

__all__ = [
    "solve_eo_preconditioned",
]
