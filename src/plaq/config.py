"""Global configuration for plaq.

This module provides global configuration settings for the plaq library,
including default data types for tensor operations.

The default dtype is :obj:`torch.complex128`, which is suitable for
high-precision lattice gauge theory computations. For example, the
Wilson action is computed as:

.. math::

    S_W = \\beta \\sum_{x, \\mu < \\nu} \\left(1 - \\frac{1}{N_c} \\text{Re} \\, \\text{Tr} \\, U_{\\mu\\nu}(x)\\right)

where :math:`U_{\\mu\\nu}(x)` is the plaquette at site :math:`x` in the
:math:`\\mu`-:math:`\\nu` plane.

Example
-------
>>> import plaq as pq
>>> print(pq.config.DEFAULT_DTYPE)
torch.complex128

>>> # Change the default dtype
>>> pq.config.DEFAULT_DTYPE = torch.complex64

"""

from typing import Any

import torch


class PlaqConfig:
    """Global configuration class for plaq.

    This class holds global settings that affect the behavior of the library.

    Attributes
    ----------
    DEFAULT_DTYPE : torch.dtype
        The default complex data type for tensor operations.
        Defaults to :obj:`torch.complex128` for high-precision computations.

    DEFAULT_DEVICE : torch.device
        The default device for tensor operations.
        Defaults to CPU.

    Notes
    -----
    The choice of :obj:`torch.complex128` as the default dtype ensures
    numerical stability in gauge field computations, where high precision
    is often required for:

    - Computing the trace of SU(N) matrices
    - Evaluating the Wilson action
    - Monte Carlo updates

    For performance-critical applications where lower precision is acceptable,
    consider using :obj:`torch.complex64`.

    """

    def __init__(self) -> None:
        """Initialize the configuration with default values."""
        self.DEFAULT_DTYPE: Any = torch.complex128
        self.DEFAULT_DEVICE: Any = torch.device("cpu")

    def reset(self) -> None:
        """Reset configuration to default values.

        This method restores all configuration options to their original
        default values.

        Example
        -------
        >>> import plaq as pq
        >>> pq.config.DEFAULT_DTYPE = torch.complex64
        >>> pq.config.reset()
        >>> print(pq.config.DEFAULT_DTYPE)
        torch.complex128

        """
        self.DEFAULT_DTYPE = torch.complex128
        self.DEFAULT_DEVICE = torch.device("cpu")

    def __repr__(self) -> str:
        """Return a string representation of the configuration."""
        return (
            f"PlaqConfig(\n"
            f"    DEFAULT_DTYPE={self.DEFAULT_DTYPE},\n"
            f"    DEFAULT_DEVICE={self.DEFAULT_DEVICE}\n"
            f")"
        )


# Global configuration instance
config = PlaqConfig()
