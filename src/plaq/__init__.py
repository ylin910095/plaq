"""Plaq: A lattice gauge theory toolkit for Python.

This package provides tools for lattice gauge theory computations using PyTorch.
The package can be imported as ``pq`` for convenience:

.. code-block:: python

    import plaq as pq

    # Create a lattice
    lat = pq.Lattice((4, 4, 4, 8))

    # Create fields
    U = pq.GaugeField.eye(lat)
    psi = pq.SpinorField.random(lat)

    # Apply Wilson operator
    bc = pq.BoundaryCondition()
    lat.build_neighbor_tables(bc)
    params = pq.WilsonParams(mass=0.1)
    result = pq.apply_M(U, psi, params, bc)

"""

from plaq._version import __version__
from plaq.config import PlaqConfig, config
from plaq.conventions import P_minus, P_plus, gamma, gamma5
from plaq.fields import GaugeField, SpinorField
from plaq.lattice import BoundaryCondition, Lattice
from plaq.layouts import LayoutType, pack_eo, unpack_eo
from plaq.operators import WilsonParams, apply_M, apply_Mdag, apply_MdagM
from plaq.solvers import SolverInfo, bicgstab, cg, solve

__all__ = [
    "BoundaryCondition",
    # Fields
    "GaugeField",
    # Lattice
    "Lattice",
    # Layouts
    "LayoutType",
    "P_minus",
    "P_plus",
    "PlaqConfig",
    "SolverInfo",
    "SpinorField",
    # Wilson operator
    "WilsonParams",
    # Version and config
    "__version__",
    "apply_M",
    "apply_Mdag",
    "apply_MdagM",
    # Solvers
    "bicgstab",
    "cg",
    "config",
    # Gamma matrices
    "gamma",
    "gamma5",
    "pack_eo",
    "solve",
    "unpack_eo",
]
