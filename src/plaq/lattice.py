"""Lattice geometry and boundary conditions for 4D lattice QCD.

This module provides the :class:`Lattice` class for managing lattice geometry,
site indexing, and neighbor lookups, as well as :class:`BoundaryCondition` for
specifying fermion boundary conditions.

The lattice uses lexicographic ordering for site indices:

.. math::

    \\text{site\\_id} = x + N_x (y + N_y (z + N_z t))

where :math:`(x, y, z, t)` are the site coordinates and :math:`(N_x, N_y, N_z, N_t)`
is the lattice shape.

Example
-------
>>> import plaq as pq
>>> lat = pq.Lattice((4, 4, 4, 8))
>>> print(lat.volume)  # 512
>>> site_id = lat.index(1, 2, 3, 4)
>>> coords = lat.coord(site_id)  # (1, 2, 3, 4)

"""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class BoundaryCondition:
    """Boundary conditions for fermion fields.

    Specifies the phase factors applied when fermion fields cross lattice
    boundaries. The standard choice for QCD is antiperiodic in time
    (to project onto finite temperature) and periodic in space.

    Attributes
    ----------
    fermion_bc_time : float
        Phase factor for temporal boundary. Default is -1 (antiperiodic).
    fermion_bc_space : float
        Phase factor for spatial boundaries. Default is +1 (periodic).

    Notes
    -----
    The boundary condition phase is applied when a fermion field value is
    accessed across the boundary:

    .. math::

        \\psi(x + L_\\mu \\hat{\\mu}) = e^{i\\theta_\\mu} \\psi(x)

    where :math:`\\theta_\\mu = 0` for periodic (+1) and :math:`\\theta_\\mu = \\pi`
    for antiperiodic (-1).

    Example
    -------
    >>> bc = BoundaryCondition()  # Default: antiperiodic in time
    >>> bc_periodic = BoundaryCondition(fermion_bc_time=1.0)  # Fully periodic

    """

    fermion_bc_time: float = -1.0
    fermion_bc_space: float = 1.0

    def get_bc_phase(self, mu: int) -> float:
        """Get the boundary condition phase for direction mu.

        Parameters
        ----------
        mu : int
            Direction index: 0, 1, 2 for space, 3 for time.

        Returns
        -------
        float
            The boundary phase (+1 or -1).

        """
        if mu == 3:
            return self.fermion_bc_time
        return self.fermion_bc_space


@dataclass
class Lattice:
    """4D lattice geometry with indexing and neighbor tables.

    Manages the geometry of a 4D hypercubic lattice with shape
    :math:`(N_x, N_y, N_z, N_t)`. Provides site indexing in lexicographic order
    and precomputed neighbor tables with boundary condition phases.

    Attributes
    ----------
    shape : tuple[int, int, int, int]
        Lattice dimensions (Nx, Ny, Nz, Nt).
    volume : int
        Total number of lattice sites.
    dtype : torch.dtype
        Default data type for field operations.
    device : torch.device
        Default device for tensor operations.

    Notes
    -----
    Site indices follow lexicographic ordering:

    .. math::

        \\text{site\\_id} = x + N_x (y + N_y (z + N_z t))

    The parity of a site is defined as:

    .. math::

        \\text{parity}(x, y, z, t) = (x + y + z + t) \\mod 2

    Example
    -------
    >>> lat = Lattice((4, 4, 4, 8))
    >>> lat.volume
    512
    >>> lat.index(1, 2, 3, 4)
    209
    >>> lat.coord(209)
    (1, 2, 3, 4)
    >>> lat.parity(209)
    0

    """

    shape: tuple[int, int, int, int]
    dtype: Any = field(default=torch.complex128)
    device: Any = field(default_factory=lambda: torch.device("cpu"))

    # Computed attributes (set in __post_init__)
    _volume: int = field(init=False, repr=False)
    _strides: tuple[int, int, int, int] = field(init=False, repr=False)
    _even_sites: torch.Tensor | None = field(init=False, repr=False, default=None)
    _odd_sites: torch.Tensor | None = field(init=False, repr=False, default=None)
    _neighbor_fwd: list[tuple[torch.Tensor, torch.Tensor]] | None = field(
        init=False, repr=False, default=None
    )
    _neighbor_bwd: list[tuple[torch.Tensor, torch.Tensor]] | None = field(
        init=False, repr=False, default=None
    )

    def __post_init__(self) -> None:
        """Compute derived quantities after initialization."""
        nx, ny, nz, nt = self.shape
        self._volume = nx * ny * nz * nt
        # Strides for lexicographic indexing: site = x + Nx*(y + Ny*(z + Nz*t))
        self._strides = (1, nx, nx * ny, nx * ny * nz)

    @property
    def volume(self) -> int:
        """Total number of lattice sites."""
        return self._volume

    @property
    def nx(self) -> int:
        """Size in x-direction."""
        return self.shape[0]

    @property
    def ny(self) -> int:
        """Size in y-direction."""
        return self.shape[1]

    @property
    def nz(self) -> int:
        """Size in z-direction."""
        return self.shape[2]

    @property
    def nt(self) -> int:
        """Size in t-direction."""
        return self.shape[3]

    def index(self, x: int, y: int, z: int, t: int) -> int:
        """Convert coordinates to site index.

        Uses lexicographic ordering:
        :math:`\\text{site} = x + N_x (y + N_y (z + N_z t))`

        Parameters
        ----------
        x, y, z, t : int
            Site coordinates.

        Returns
        -------
        int
            Site index in range [0, volume).

        """
        return x + self._strides[1] * y + self._strides[2] * z + self._strides[3] * t

    def coord(self, site_id: int) -> tuple[int, int, int, int]:
        """Convert site index to coordinates.

        Parameters
        ----------
        site_id : int
            Site index in range [0, volume).

        Returns
        -------
        tuple[int, int, int, int]
            Coordinates (x, y, z, t).

        """
        nx, ny, nz, _nt = self.shape
        t, rem = divmod(site_id, nx * ny * nz)
        z, rem = divmod(rem, nx * ny)
        y, x = divmod(rem, nx)
        return (x, y, z, t)

    def coord_tensor(self, site_ids: torch.Tensor) -> torch.Tensor:
        """Convert site index tensor to coordinate tensor.

        Parameters
        ----------
        site_ids : torch.Tensor
            Tensor of site indices.

        Returns
        -------
        torch.Tensor
            Tensor of shape (..., 4) with coordinates [x, y, z, t].

        """
        nx, ny, nz, _nt = self.shape
        nxny = nx * ny
        nxnynz = nxny * nz

        t = site_ids // nxnynz
        rem = site_ids % nxnynz
        z = rem // nxny
        rem = rem % nxny
        y = rem // nx
        x = rem % nx

        return torch.stack([x, y, z, t], dim=-1)

    def index_tensor(self, coords: torch.Tensor) -> torch.Tensor:
        """Convert coordinate tensor to site index tensor.

        Parameters
        ----------
        coords : torch.Tensor
            Tensor of shape (..., 4) with coordinates [x, y, z, t].

        Returns
        -------
        torch.Tensor
            Tensor of site indices.

        """
        x = coords[..., 0]
        y = coords[..., 1]
        z = coords[..., 2]
        t = coords[..., 3]
        return x + self._strides[1] * y + self._strides[2] * z + self._strides[3] * t

    def parity(self, site_id: int) -> int:
        """Compute the parity (even=0, odd=1) of a site.

        The parity is defined as :math:`(x + y + z + t) \\mod 2`.

        Parameters
        ----------
        site_id : int
            Site index.

        Returns
        -------
        int
            0 for even sites, 1 for odd sites.

        """
        x, y, z, t = self.coord(site_id)
        return (x + y + z + t) % 2

    def parity_tensor(self, site_ids: torch.Tensor) -> torch.Tensor:
        """Compute parities for a tensor of site indices.

        Parameters
        ----------
        site_ids : torch.Tensor
            Tensor of site indices.

        Returns
        -------
        torch.Tensor
            Tensor of parities (0 or 1).

        """
        coords = self.coord_tensor(site_ids)
        return (coords.sum(dim=-1)) % 2

    @property
    def even_sites(self) -> torch.Tensor:
        """Indices of even-parity sites in lexicographic order.

        Returns
        -------
        torch.Tensor
            LongTensor of even site indices.

        """
        if self._even_sites is None:
            self._compute_parity_sites()
        assert self._even_sites is not None
        return self._even_sites

    @property
    def odd_sites(self) -> torch.Tensor:
        """Indices of odd-parity sites in lexicographic order.

        Returns
        -------
        torch.Tensor
            LongTensor of odd site indices.

        """
        if self._odd_sites is None:
            self._compute_parity_sites()
        assert self._odd_sites is not None
        return self._odd_sites

    def _compute_parity_sites(self) -> None:
        """Compute and cache even/odd site index lists."""
        all_sites = torch.arange(self._volume, device=self.device)
        parities = self.parity_tensor(all_sites)
        self._even_sites = all_sites[parities == 0]
        self._odd_sites = all_sites[parities == 1]

    def build_neighbor_tables(self, bc: BoundaryCondition) -> None:
        """Build neighbor lookup tables with boundary condition phases.

        Computes forward and backward neighbor indices for all sites in all
        directions, along with the boundary condition phase factors.

        Parameters
        ----------
        bc : BoundaryCondition
            Boundary conditions to apply.

        Notes
        -----
        After calling this method, neighbor lookups are available via:

        - ``neighbor_fwd(mu)`` returns (neighbor_indices, bc_phases)
        - ``neighbor_bwd(mu)`` returns (neighbor_indices, bc_phases)

        """
        device = self.device
        all_sites = torch.arange(self._volume, dtype=torch.long, device=device)
        coords = self.coord_tensor(all_sites)  # [V, 4]

        neighbor_fwd: list[tuple[torch.Tensor, torch.Tensor]] = []
        neighbor_bwd: list[tuple[torch.Tensor, torch.Tensor]] = []

        for mu in range(4):
            # Forward neighbor: x + hat{mu}
            fwd_coords = coords.clone()
            fwd_coords[:, mu] = (fwd_coords[:, mu] + 1) % self.shape[mu]
            fwd_indices = self.index_tensor(fwd_coords)

            # Backward neighbor: x - hat{mu}
            bwd_coords = coords.clone()
            bwd_coords[:, mu] = (bwd_coords[:, mu] - 1) % self.shape[mu]
            bwd_indices = self.index_tensor(bwd_coords)

            # Boundary phases
            bc_phase = bc.get_bc_phase(mu)
            fwd_phases = torch.ones(self._volume, dtype=self.dtype, device=device)
            bwd_phases = torch.ones(self._volume, dtype=self.dtype, device=device)

            # Forward: phase applies when crossing upper boundary
            fwd_phases[coords[:, mu] == self.shape[mu] - 1] = bc_phase

            # Backward: phase applies when crossing lower boundary
            bwd_phases[coords[:, mu] == 0] = bc_phase

            neighbor_fwd.append((fwd_indices, fwd_phases))
            neighbor_bwd.append((bwd_indices, bwd_phases))

        self._neighbor_fwd = neighbor_fwd
        self._neighbor_bwd = neighbor_bwd

    def neighbor_fwd(self, mu: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get forward neighbor indices and BC phases for direction mu.

        Parameters
        ----------
        mu : int
            Direction index (0-3).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (neighbor_indices, bc_phases) where neighbor_indices[site] gives
            the index of the neighbor at site+hat{mu}, and bc_phases[site]
            gives the boundary phase factor.

        Raises
        ------
        RuntimeError
            If neighbor tables have not been built.

        """
        if self._neighbor_fwd is None:
            msg = "Neighbor tables not built. Call build_neighbor_tables(bc) first."
            raise RuntimeError(msg)
        return self._neighbor_fwd[mu]

    def neighbor_bwd(self, mu: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get backward neighbor indices and BC phases for direction mu.

        Parameters
        ----------
        mu : int
            Direction index (0-3).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (neighbor_indices, bc_phases) where neighbor_indices[site] gives
            the index of the neighbor at site-hat{mu}, and bc_phases[site]
            gives the boundary phase factor.

        Raises
        ------
        RuntimeError
            If neighbor tables have not been built.

        """
        if self._neighbor_bwd is None:
            msg = "Neighbor tables not built. Call build_neighbor_tables(bc) first."
            raise RuntimeError(msg)
        return self._neighbor_bwd[mu]
