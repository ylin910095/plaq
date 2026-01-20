"""Field containers for lattice QCD.

This module provides layout-aware field containers for lattice QCD computations:

- :class:`SpinorField`: Fermion field with spin and color indices
- :class:`GaugeField`: Gauge field (link variables) in SU(3)

Both field types support the "site" and "eo" (even-odd) layouts, with
automatic packing/unpacking and caching.

Example
-------
>>> import plaq as pq
>>> import torch
>>> lat = pq.Lattice((4, 4, 4, 8))
>>>
>>> # Create a random spinor field
>>> psi = pq.SpinorField.random(lat)
>>> print(psi.site.shape)  # [512, 4, 3]
>>>
>>> # Convert to even-odd layout
>>> psi_eo = psi.as_layout("eo")
>>> print(psi_eo.eo.shape)  # [2, 256, 4, 3]
>>>
>>> # Create identity gauge field
>>> U = pq.GaugeField.eye(lat)
>>> print(U.data.shape)  # [4, 512, 3, 3]

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from plaq.layouts import LayoutType, pack_eo, unpack_eo

if TYPE_CHECKING:
    from plaq.lattice import Lattice


class SpinorField:
    """Fermion spinor field on a lattice.

    A spinor field :math:`\\psi(x)` has spin index :math:`\\alpha \\in \\{0,1,2,3\\}`
    and color index :math:`a \\in \\{0,1,2\\}` at each lattice site :math:`x`.

    The field supports two storage layouts:

    - **site**: Shape ``[V, 4, 3]`` - canonical user-facing layout
    - **eo**: Shape ``[2, V/2, 4, 3]`` - even-odd checkerboard layout

    The field caches conversions between layouts for efficiency. Fields are
    treated as immutable; use :meth:`clone` to create a modifiable copy.

    Attributes
    ----------
    lattice : Lattice
        The lattice geometry.
    dtype : torch.dtype
        Data type of the field.
    device : torch.device
        Device where the field is stored.
    layout : LayoutType
        Current primary layout ("site" or "eo").

    Notes
    -----
    In the Dirac basis, the spinor has four components corresponding to
    particle/antiparticle and spin up/down degrees of freedom.

    Example
    -------
    >>> lat = Lattice((4, 4, 4, 8))
    >>> psi = SpinorField.random(lat)
    >>> print(psi.site.shape)  # [512, 4, 3]
    >>> print(psi.eo.shape)  # [2, 256, 4, 3]

    """

    def __init__(
        self,
        data: torch.Tensor,
        lattice: Lattice,
        layout: LayoutType = "site",
    ) -> None:
        """Initialize a spinor field.

        Parameters
        ----------
        data : torch.Tensor
            Field data tensor. Shape depends on layout:
            - "site": [V, 4, 3]
            - "eo": [2, V/2, 4, 3]
        lattice : Lattice
            Lattice geometry.
        layout : LayoutType
            Layout of the input data.

        """
        self._data = data
        self._lattice = lattice
        self._layout: LayoutType = layout
        self._dtype = data.dtype
        self._device = data.device

        # Cached representations
        self._site_cache: torch.Tensor | None = None
        self._eo_cache: torch.Tensor | None = None

        # Set initial cache
        if layout == "site":
            self._site_cache = data
        else:
            self._eo_cache = data

    @property
    def lattice(self) -> Lattice:
        """The lattice geometry."""
        return self._lattice

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the field."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        """Device where the field is stored."""
        return self._device

    @property
    def layout(self) -> LayoutType:
        """Current primary layout."""
        return self._layout

    @property
    def data(self) -> torch.Tensor:
        """Raw underlying data tensor in the primary layout."""
        return self._data

    @property
    def site(self) -> torch.Tensor:
        """Field data in site layout [V, 4, 3].

        Returns a view or cached conversion. If the field is stored in
        eo layout, this will unpack and cache the result.

        """
        if self._site_cache is not None:
            return self._site_cache

        # Need to unpack from eo
        assert self._eo_cache is not None
        self._site_cache = unpack_eo(self._eo_cache, self._lattice)
        return self._site_cache

    @property
    def eo(self) -> torch.Tensor:
        """Field data in even-odd layout [2, V/2, 4, 3].

        Returns a view or cached conversion. If the field is stored in
        site layout, this will pack and cache the result.

        """
        if self._eo_cache is not None:
            return self._eo_cache

        # Need to pack from site
        assert self._site_cache is not None
        self._eo_cache = pack_eo(self._site_cache, self._lattice)
        return self._eo_cache

    def as_layout(self, layout: LayoutType) -> SpinorField:
        """Return a SpinorField with the specified primary layout.

        Parameters
        ----------
        layout : LayoutType
            Target layout ("site" or "eo").

        Returns
        -------
        SpinorField
            New SpinorField with the requested layout.
            If already in the target layout, may return self.

        """
        if layout == self._layout:
            return self

        data = self.site if layout == "site" else self.eo

        return SpinorField(data, self._lattice, layout=layout)

    def clone(self) -> SpinorField:
        """Create a deep copy of this field.

        Returns
        -------
        SpinorField
            A new SpinorField with cloned data.

        """
        return SpinorField(
            self._data.clone(),
            self._lattice,
            layout=self._layout,
        )

    @classmethod
    def zeros(
        cls,
        lattice: Lattice,
        dtype: Any = None,
        device: Any = None,
        layout: LayoutType = "site",
    ) -> SpinorField:
        """Create a zero-initialized spinor field.

        Parameters
        ----------
        lattice : Lattice
            Lattice geometry.
        dtype : torch.dtype, optional
            Data type. Defaults to lattice.dtype.
        device : torch.device, optional
            Device. Defaults to lattice.device.
        layout : LayoutType
            Storage layout.

        Returns
        -------
        SpinorField
            Zero-initialized field.

        """
        if dtype is None:
            dtype = lattice.dtype
        if device is None:
            device = lattice.device

        shape = (lattice.volume, 4, 3) if layout == "site" else (2, lattice.volume // 2, 4, 3)

        data = torch.zeros(shape, dtype=dtype, device=device)
        return cls(data, lattice, layout=layout)

    @classmethod
    def random(
        cls,
        lattice: Lattice,
        dtype: Any = None,
        device: Any = None,
        layout: LayoutType = "site",
    ) -> SpinorField:
        """Create a random spinor field.

        Parameters
        ----------
        lattice : Lattice
            Lattice geometry.
        dtype : torch.dtype, optional
            Data type. Defaults to lattice.dtype.
        device : torch.device, optional
            Device. Defaults to lattice.device.
        layout : LayoutType
            Storage layout.

        Returns
        -------
        SpinorField
            Random complex field with standard normal distribution.

        """
        if dtype is None:
            dtype = lattice.dtype
        if device is None:
            device = lattice.device

        shape = (lattice.volume, 4, 3) if layout == "site" else (2, lattice.volume // 2, 4, 3)

        # Complex random: real and imaginary parts from standard normal
        data = torch.randn(shape, dtype=dtype, device=device)
        return cls(data, lattice, layout=layout)


class GaugeField:
    """Gauge field (link variables) on a lattice.

    A gauge field consists of SU(3) matrices :math:`U_\\mu(x)` associated with
    each link of the lattice, connecting site :math:`x` to site :math:`x + \\hat{\\mu}`.

    Currently only site layout is supported, with shape ``[4, V, 3, 3]`` where:
    - First index is the direction :math:`\\mu \\in \\{0, 1, 2, 3\\}`
    - Second index is the site
    - Last two indices are the 3x3 color matrix

    Attributes
    ----------
    lattice : Lattice
        The lattice geometry.
    dtype : torch.dtype
        Data type of the field.
    device : torch.device
        Device where the field is stored.

    Notes
    -----
    The gauge field transforms under local gauge transformations as:

    .. math::

        U_\\mu(x) \\to \\Omega(x) U_\\mu(x) \\Omega^\\dagger(x + \\hat{\\mu})

    where :math:`\\Omega(x) \\in SU(3)` is the gauge transformation at site :math:`x`.

    Example
    -------
    >>> lat = Lattice((4, 4, 4, 8))
    >>> U = GaugeField.eye(lat)
    >>> print(U.data.shape)  # [4, 512, 3, 3]

    """

    def __init__(
        self,
        data: torch.Tensor,
        lattice: Lattice,
    ) -> None:
        """Initialize a gauge field.

        Parameters
        ----------
        data : torch.Tensor
            Field data tensor with shape [4, V, 3, 3].
        lattice : Lattice
            Lattice geometry.

        """
        self._data = data
        self._lattice = lattice
        self._dtype = data.dtype
        self._device = data.device

    @property
    def lattice(self) -> Lattice:
        """The lattice geometry."""
        return self._lattice

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the field."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        """Device where the field is stored."""
        return self._device

    @property
    def data(self) -> torch.Tensor:
        """Raw field data tensor [4, V, 3, 3]."""
        return self._data

    def __getitem__(self, mu: int) -> torch.Tensor:
        """Get gauge links in direction mu.

        Parameters
        ----------
        mu : int
            Direction index (0-3).

        Returns
        -------
        torch.Tensor
            Tensor of shape [V, 3, 3] containing link matrices.

        """
        return self._data[mu]

    @classmethod
    def eye(
        cls,
        lattice: Lattice,
        dtype: Any = None,
        device: Any = None,
    ) -> GaugeField:
        """Create an identity (unit) gauge field.

        All link variables are set to the 3x3 identity matrix.

        Parameters
        ----------
        lattice : Lattice
            Lattice geometry.
        dtype : torch.dtype, optional
            Data type. Defaults to lattice.dtype.
        device : torch.device, optional
            Device. Defaults to lattice.device.

        Returns
        -------
        GaugeField
            Identity gauge field.

        """
        if dtype is None:
            dtype = lattice.dtype
        if device is None:
            device = lattice.device

        # Create [4, V, 3, 3] tensor of identity matrices
        data = torch.zeros(4, lattice.volume, 3, 3, dtype=dtype, device=device)
        eye = torch.eye(3, dtype=dtype, device=device)
        for mu in range(4):
            data[mu, :, :, :] = eye

        return cls(data, lattice)

    @classmethod
    def random(
        cls,
        lattice: Lattice,
        dtype: Any = None,
        device: Any = None,
    ) -> GaugeField:
        """Create a random SU(3) gauge field sampled from the Haar measure.

        Uses the standard algorithm for sampling uniform random unitary matrices:
        1. Generate a random complex Gaussian matrix Z
        2. Compute the QR decomposition: Z = Q R
        3. Fix the signs of R's diagonal to make Q unique
        4. Project to SU(3) by dividing by det(Q)^(1/3)

        This produces matrices uniformly distributed according to the
        Haar measure on SU(3).

        Parameters
        ----------
        lattice : Lattice
            Lattice geometry.
        dtype : torch.dtype, optional
            Data type. Defaults to lattice.dtype.
        device : torch.device, optional
            Device. Defaults to lattice.device.

        Returns
        -------
        GaugeField
            Random SU(3) gauge field sampled from Haar measure.

        References
        ----------
        .. [1] Mezzadri, F. (2007). "How to generate random matrices from the
               classical compact groups." Notices of the AMS 54.5: 592-604.
               https://arxiv.org/abs/math-ph/0609050

        """
        if dtype is None:
            dtype = lattice.dtype
        if device is None:
            device = lattice.device

        shape = (4, lattice.volume, 3, 3)

        # Generate random complex Gaussian matrices
        # Real and imaginary parts are independent standard normal
        Z_real = torch.randn(shape, dtype=torch.float64, device=device)
        Z_imag = torch.randn(shape, dtype=torch.float64, device=device)
        Z = torch.complex(Z_real, Z_imag)

        # QR decomposition: Z = Q @ R
        # torch.linalg.qr returns Q with shape [..., 3, 3]
        Q, R = torch.linalg.qr(Z)

        # Fix the phase ambiguity: make diagonal of R positive real
        # This ensures Q is uniquely determined and uniformly distributed
        diag_R = torch.diagonal(R, dim1=-2, dim2=-1)  # [..., 3]
        diag_phases = diag_R / torch.abs(diag_R)  # Unit complex numbers
        # Multiply each column of Q by the conjugate of the corresponding phase
        Q = Q * diag_phases.unsqueeze(-2).conj()

        # Now Q is a random unitary matrix (Haar-distributed on U(3))
        # Project to SU(3) by dividing by det(Q)^(1/3)
        det_Q = torch.linalg.det(Q)  # [...] complex
        # Take cube root of determinant: det^(1/3) = |det|^(1/3) * exp(i*arg/3)
        det_phase = det_Q / torch.abs(det_Q)  # Unit complex number
        det_phase_third = torch.pow(det_phase, 1.0 / 3.0)  # Phase^(1/3)

        # Divide all elements by det^(1/3) to get det = 1
        Q = Q / det_phase_third.unsqueeze(-1).unsqueeze(-1)

        # Convert to target dtype
        data = Q.to(dtype=dtype)

        return cls(data, lattice)

    def clone(self) -> GaugeField:
        """Create a deep copy of this field.

        Returns
        -------
        GaugeField
            A new GaugeField with cloned data.

        """
        return GaugeField(self._data.clone(), self._lattice)
