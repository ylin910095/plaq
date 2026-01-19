"""Operators subpackage for plaq.

This subpackage provides lattice QCD operators, including the Wilson
Dirac operator.

"""

from plaq.operators.wilson import (
    WilsonParams,
    apply_M,
    apply_Mdag,
    apply_MdagM,
)

__all__ = [
    "WilsonParams",
    "apply_M",
    "apply_Mdag",
    "apply_MdagM",
]
