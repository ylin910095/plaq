"""Solvers subpackage for plaq.

This subpackage provides the high-level solver API for lattice QCD linear systems:

- :func:`solve`: High-level API that automatically selects solver and equation type
- :class:`SolverInfo`: Dataclass containing solver convergence information

The low-level Krylov solver implementations (CG, BiCGStab) are located in the
backend modules (see :mod:`plaq.backends.plaq`). They are re-exported here
for convenience:

- :func:`cg`: Conjugate Gradient for Hermitian positive-definite systems (e.g., MdagM)
- :func:`bicgstab`: BiCGStab for general non-Hermitian systems (e.g., M)

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

from plaq.backends.plaq.bicgstab import bicgstab
from plaq.backends.plaq.cg import cg
from plaq.solvers.api import SolverInfo, solve

__all__ = [
    "SolverInfo",
    "bicgstab",
    "cg",
    "solve",
]
