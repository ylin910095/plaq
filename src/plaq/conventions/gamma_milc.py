"""MILC gamma matrix convention for 4D Euclidean lattice QCD.

This module provides the gamma matrices in the MILC convention commonly used
in lattice QCD simulations. All matrices are 4x4 complex tensors.

The gamma matrices satisfy the Clifford algebra:

.. math::

    \\{\\gamma_\\mu, \\gamma_\\nu\\} = 2\\delta_{\\mu\\nu} I

where :math:`\\mu, \\nu \\in \\{0, 1, 2, 3\\}` correspond to the x, y, z, t directions.

The chiral matrix :math:`\\gamma_5` is defined as:

.. math::

    \\gamma_5 = \\gamma_0 \\gamma_1 \\gamma_2 \\gamma_3

and satisfies :math:`\\gamma_5^2 = I` and :math:`\\{\\gamma_5, \\gamma_\\mu\\} = 0`.

Example
-------
>>> import plaq as pq
>>> # Access gamma matrices
>>> gamma_0 = pq.gamma[0]
>>> gamma5 = pq.gamma5
>>> # Compute projectors
>>> P_plus_0 = pq.P_plus(0)  # (I + gamma_0) / 2
>>> P_minus_0 = pq.P_minus(0)  # (I - gamma_0) / 2

"""

from typing import Any

import torch

# Default dtype for gamma matrices
_GAMMA_DTYPE = torch.complex128


def _build_gamma_matrices(
    dtype: torch.dtype = _GAMMA_DTYPE, device: torch.device | None = None
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Build the MILC gamma matrices.

    The MILC convention uses the following explicit representation:

    .. math::

        \\gamma_0 = \\begin{pmatrix} 0 & 0 & 0 & i \\\\
                                     0 & 0 & i & 0 \\\\
                                     0 & -i & 0 & 0 \\\\
                                     -i & 0 & 0 & 0 \\end{pmatrix}

        \\gamma_1 = \\begin{pmatrix} 0 & 0 & 0 & -1 \\\\
                                     0 & 0 & 1 & 0 \\\\
                                     0 & 1 & 0 & 0 \\\\
                                     -1 & 0 & 0 & 0 \\end{pmatrix}

        \\gamma_2 = \\begin{pmatrix} 0 & 0 & i & 0 \\\\
                                     0 & 0 & 0 & -i \\\\
                                     -i & 0 & 0 & 0 \\\\
                                     0 & i & 0 & 0 \\end{pmatrix}

        \\gamma_3 = \\begin{pmatrix} 0 & 0 & 1 & 0 \\\\
                                     0 & 0 & 0 & 1 \\\\
                                     1 & 0 & 0 & 0 \\\\
                                     0 & 1 & 0 & 0 \\end{pmatrix}

    Parameters
    ----------
    dtype : torch.dtype
        Data type for the matrices. Default is torch.complex128.
    device : torch.device, optional
        Device for the matrices. Default is CPU.

    Returns
    -------
    tuple[list[torch.Tensor], torch.Tensor]
        A tuple (gamma_list, gamma5) where gamma_list[mu] is the gamma_mu matrix.

    """
    if device is None:
        device = torch.device("cpu")

    # Initialize with zeros
    g = [torch.zeros(4, 4, dtype=dtype, device=device) for _ in range(4)]

    # gamma_0 (x-direction)
    # pyre-ignore[6]: PyTorch handles complex assignment at runtime
    g[0][0, 3] = 1j
    # pyre-ignore[6]: PyTorch handles complex assignment at runtime
    g[0][1, 2] = 1j
    # pyre-ignore[6]: PyTorch handles complex assignment at runtime
    g[0][2, 1] = -1j
    # pyre-ignore[6]: PyTorch handles complex assignment at runtime
    g[0][3, 0] = -1j

    # gamma_1 (y-direction)
    g[1][0, 3] = -1.0
    g[1][1, 2] = 1.0
    g[1][2, 1] = 1.0
    g[1][3, 0] = -1.0

    # gamma_2 (z-direction)
    # pyre-ignore[6]: PyTorch handles complex assignment at runtime
    g[2][0, 2] = 1j
    # pyre-ignore[6]: PyTorch handles complex assignment at runtime
    g[2][1, 3] = -1j
    # pyre-ignore[6]: PyTorch handles complex assignment at runtime
    g[2][2, 0] = -1j
    # pyre-ignore[6]: PyTorch handles complex assignment at runtime
    g[2][3, 1] = 1j

    # gamma_3 (t-direction)
    g[3][0, 2] = 1.0
    g[3][1, 3] = 1.0
    g[3][2, 0] = 1.0
    g[3][3, 1] = 1.0

    # gamma5 = gamma_0 @ gamma_1 @ gamma_2 @ gamma_3
    g5 = g[0] @ g[1] @ g[2] @ g[3]

    return g, g5


# Build the default gamma matrices
_gamma_list, _gamma5 = _build_gamma_matrices()


class GammaMatrices:
    """Container for gamma matrices with dictionary-like access.

    This class provides access to the four gamma matrices :math:`\\gamma_\\mu`
    for :math:`\\mu \\in \\{0, 1, 2, 3\\}`.

    The gamma matrices satisfy the Clifford algebra:

    .. math::

        \\{\\gamma_\\mu, \\gamma_\\nu\\} = 2\\delta_{\\mu\\nu} I

    Example
    -------
    >>> gamma_0 = gamma[0]
    >>> gamma_3 = gamma[3]

    """

    def __init__(self, gamma_list: list[torch.Tensor]) -> None:
        """Initialize with a list of gamma matrices."""
        self._gamma = gamma_list

    def __getitem__(self, mu: int) -> torch.Tensor:
        """Get gamma_mu matrix.

        Parameters
        ----------
        mu : int
            Direction index in {0, 1, 2, 3}.

        Returns
        -------
        torch.Tensor
            The 4x4 gamma matrix for direction mu.

        Raises
        ------
        IndexError
            If mu is not in {0, 1, 2, 3}.

        """
        if not 0 <= mu <= 3:
            msg = f"Gamma index must be 0-3, got {mu}"
            raise IndexError(msg)
        return self._gamma[mu]

    def __len__(self) -> int:
        """Return the number of gamma matrices (4)."""
        return 4

    def to(
        self, dtype: torch.dtype | None = None, device: torch.device | None = None
    ) -> "GammaMatrices":
        """Return gamma matrices with specified dtype and device.

        Parameters
        ----------
        dtype : torch.dtype, optional
            Target data type.
        device : torch.device, optional
            Target device.

        Returns
        -------
        GammaMatrices
            New GammaMatrices instance with specified dtype/device.

        """
        new_gamma = [g.to(dtype=dtype, device=device) for g in self._gamma]
        return GammaMatrices(new_gamma)


# Module-level gamma matrix access
gamma: GammaMatrices = GammaMatrices(_gamma_list)
"""The four gamma matrices :math:`\\gamma_\\mu` for :math:`\\mu \\in \\{0,1,2,3\\}`."""

gamma5: torch.Tensor = _gamma5
"""The chiral gamma matrix :math:`\\gamma_5 = \\gamma_0\\gamma_1\\gamma_2\\gamma_3`."""


def get_gamma5(
    dtype: torch.dtype = _GAMMA_DTYPE, device: torch.device | None = None
) -> torch.Tensor:
    """Get gamma5 matrix with specified dtype and device.

    Parameters
    ----------
    dtype : torch.dtype
        Data type for the matrix.
    device : torch.device, optional
        Device for the matrix.

    Returns
    -------
    torch.Tensor
        The 4x4 gamma5 matrix.

    """
    _, g5 = _build_gamma_matrices(dtype=dtype, device=device)
    return g5


def P_plus(mu: int, dtype: Any = None, device: Any = None) -> torch.Tensor:
    """Compute the positive projector :math:`P_+^\\mu = (I + \\gamma_\\mu) / 2`.

    The projector satisfies :math:`P_+^\\mu + P_-^\\mu = I` and
    :math:`(P_+^\\mu)^2 = P_+^\\mu`.

    Parameters
    ----------
    mu : int
        Direction index in {0, 1, 2, 3}.
    dtype : torch.dtype, optional
        Data type. Uses default if None.
    device : torch.device, optional
        Device. Uses CPU if None.

    Returns
    -------
    torch.Tensor
        The 4x4 projector matrix.

    """
    if dtype is None:
        dtype = _GAMMA_DTYPE
    if device is None:
        device = torch.device("cpu")

    g, _ = _build_gamma_matrices(dtype=dtype, device=device)
    identity = torch.eye(4, dtype=dtype, device=device)
    return (identity + g[mu]) / 2


def P_minus(mu: int, dtype: Any = None, device: Any = None) -> torch.Tensor:
    """Compute the negative projector :math:`P_-^\\mu = (I - \\gamma_\\mu) / 2`.

    The projector satisfies :math:`P_+^\\mu + P_-^\\mu = I` and
    :math:`(P_-^\\mu)^2 = P_-^\\mu`.

    Parameters
    ----------
    mu : int
        Direction index in {0, 1, 2, 3}.
    dtype : torch.dtype, optional
        Data type. Uses default if None.
    device : torch.device, optional
        Device. Uses CPU if None.

    Returns
    -------
    torch.Tensor
        The 4x4 projector matrix.

    """
    if dtype is None:
        dtype = _GAMMA_DTYPE
    if device is None:
        device = torch.device("cpu")

    g, _ = _build_gamma_matrices(dtype=dtype, device=device)
    identity = torch.eye(4, dtype=dtype, device=device)
    return (identity - g[mu]) / 2
