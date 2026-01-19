"""Conventions subpackage for plaq.

This subpackage provides standard conventions used in lattice QCD, including
gamma matrix representations.

Currently supported conventions:

- MILC: The MILC collaboration gamma matrix convention

"""

from plaq.conventions.gamma_milc import (
    GammaMatrices,
    P_minus,
    P_plus,
    gamma,
    gamma5,
    get_gamma5,
)

__all__ = [
    "GammaMatrices",
    "P_minus",
    "P_plus",
    "gamma",
    "gamma5",
    "get_gamma5",
]
