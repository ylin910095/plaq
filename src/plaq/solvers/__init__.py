"""Solvers subpackage for plaq.

This subpackage provides iterative Krylov solvers for lattice QCD linear systems:

- :func:`cg`: Conjugate Gradient for Hermitian positive-definite systems (e.g., MdagM)
- :func:`bicgstab`: BiCGStab for general non-Hermitian systems (e.g., M)
- :func:`solve`: High-level API that automatically selects solver and equation type

Example
-------
>>> import plaq as pq
>>> lat = pq.Lattice((4, 4, 4, 8))
>>> bc = pq.BoundaryCondition()
>>> lat.build_neighbor_tables(bc)
>>> U = pq.GaugeField.eye(lat)
>>> b = pq.SpinorField.random(lat)
>>> x, info = pq.solve(U, b)
>>> print(f"Converged: {info.converged}, iters: {info.iters}")

"""

from plaq.solvers.api import SolverInfo, solve
from plaq.solvers.bicgstab import bicgstab
from plaq.solvers.cg import cg

__all__ = [
    "SolverInfo",
    "bicgstab",
    "cg",
    "solve",
]
