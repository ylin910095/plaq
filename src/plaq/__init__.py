"""Plaq: A lattice gauge theory toolkit for Python.

This package provides tools for lattice gauge theory computations using PyTorch.
The package can be imported as ``pq`` for convenience:

.. code-block:: python

    import plaq as pq

    # Access default configuration
    print(pq.config.DEFAULT_DTYPE)

"""

from plaq._version import __version__
from plaq.config import PlaqConfig, config

__all__ = [
    "PlaqConfig",
    "__version__",
    "config",
]
