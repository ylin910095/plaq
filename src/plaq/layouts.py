"""Layout system for lattice field storage.

This module provides support for different data layouts used in lattice QCD:

- **site** layout: Canonical user-facing layout with site-major ordering.
  Tensor shape for spinor: ``[V, 4, 3]`` where V is the lattice volume.

- **eo** (even-odd) layout: Checkerboard layout for efficient operator
  implementations. Tensor shape for spinor: ``[2, V/2, 4, 3]`` where
  index 0 contains even sites and index 1 contains odd sites.

The even-odd (checkerboard) decomposition is based on site parity:

.. math::

    \\text{parity}(x, y, z, t) = (x + y + z + t) \\mod 2

Sites with parity 0 are "even" and sites with parity 1 are "odd".

Example
-------
>>> import plaq as pq
>>> lat = pq.Lattice((4, 4, 4, 8))
>>> # Create a random spinor in site layout
>>> psi_site = torch.randn(lat.volume, 4, 3, dtype=torch.complex128)
>>> # Pack to even-odd layout
>>> psi_eo = pq.pack_eo(psi_site, lat)
>>> print(psi_eo.shape)  # torch.Size([2, 256, 4, 3])
>>> # Unpack back to site layout
>>> psi_site2 = pq.unpack_eo(psi_eo, lat)
>>> torch.allclose(psi_site, psi_site2)  # True

"""

from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from plaq.lattice import Lattice

# Layout type alias
LayoutType = Literal["site", "eo"]


def pack_eo(site_tensor: torch.Tensor, lattice: "Lattice") -> torch.Tensor:
    """Pack a site-layout tensor into even-odd (checkerboard) layout.

    Converts a tensor from site-major ordering to even-odd ordering where
    even sites (parity 0) come first, followed by odd sites (parity 1).

    Parameters
    ----------
    site_tensor : torch.Tensor
        Input tensor with shape ``[V, ...]`` where V is the lattice volume.
    lattice : Lattice
        Lattice object providing geometry and parity information.

    Returns
    -------
    torch.Tensor
        Output tensor with shape ``[2, V/2, ...]`` where index 0 contains
        even sites and index 1 contains odd sites.

    Notes
    -----
    The packing is deterministic: even sites are stored in their relative
    lexicographic order within the parity class, and similarly for odd sites.

    .. math::

        \\psi_{\\text{eo}}[0, i, :] = \\psi_{\\text{site}}[\\text{even\\_sites}[i], :]
        \\psi_{\\text{eo}}[1, j, :] = \\psi_{\\text{site}}[\\text{odd\\_sites}[j], :]

    """
    even_sites = lattice.even_sites
    odd_sites = lattice.odd_sites

    # Gather by parity
    even_data = site_tensor[even_sites]  # [V/2, ...]
    odd_data = site_tensor[odd_sites]  # [V/2, ...]

    # Stack: [2, V/2, ...]
    return torch.stack([even_data, odd_data], dim=0)


def unpack_eo(eo_tensor: torch.Tensor, lattice: "Lattice") -> torch.Tensor:
    """Unpack an even-odd layout tensor to site layout.

    Converts a tensor from even-odd ordering back to site-major ordering.

    Parameters
    ----------
    eo_tensor : torch.Tensor
        Input tensor with shape ``[2, V/2, ...]`` where index 0 contains
        even sites and index 1 contains odd sites.
    lattice : Lattice
        Lattice object providing geometry and parity information.

    Returns
    -------
    torch.Tensor
        Output tensor with shape ``[V, ...]`` in site-major order.

    Notes
    -----
    This is the inverse of :func:`pack_eo`. The unpacking restores the
    original site ordering:

    .. math::

        \\psi_{\\text{site}}[\\text{even\\_sites}[i], :] = \\psi_{\\text{eo}}[0, i, :]
        \\psi_{\\text{site}}[\\text{odd\\_sites}[j], :] = \\psi_{\\text{eo}}[1, j, :]

    """
    even_sites = lattice.even_sites
    odd_sites = lattice.odd_sites

    # Get remaining dimensions from eo_tensor
    remaining_shape = eo_tensor.shape[2:]
    output_shape = (lattice.volume, *remaining_shape)

    # Allocate output
    site_tensor = torch.empty(output_shape, dtype=eo_tensor.dtype, device=eo_tensor.device)

    # Scatter by parity
    site_tensor[even_sites] = eo_tensor[0]
    site_tensor[odd_sites] = eo_tensor[1]

    return site_tensor
