"""Wilson Dirac operator for lattice QCD.

This module implements the naive Wilson Dirac operator with user-specified
boundary conditions. The implementation operates in site layout as the
canonical reference and supports both site and even-odd input layouts.

The Wilson Dirac operator is defined as:

.. math::

    M\\psi(x) = (m_0 + 4r)\\psi(x)
    - \\frac{1}{2}\\sum_{\\mu=0}^{3} \\left[
        (r - \\gamma_\\mu) U_\\mu(x) \\psi(x+\\hat{\\mu})
        + (r + \\gamma_\\mu) U_\\mu^\\dagger(x-\\hat{\\mu}) \\psi(x-\\hat{\\mu})
    \\right]

where:

- :math:`m_0` is the bare mass parameter
- :math:`r` is the Wilson parameter (fixed to 1.0)
- :math:`U_\\mu(x)` are the gauge links
- :math:`\\gamma_\\mu` are the Dirac gamma matrices

The Wilson parameter :math:`r` lifts the fermion doublers by adding a
momentum-dependent mass term that vanishes for physical modes at the
origin of the Brillouin zone.

The relationship to the hopping parameter :math:`\\kappa` is:

.. math::

    \\kappa = \\frac{1}{2(m_0 + 4r)}

Notes
-----
The operator satisfies :math:`\\gamma_5`-hermiticity:

.. math::

    M^\\dagger = \\gamma_5 M \\gamma_5

Example
-------
>>> import plaq as pq
>>> lat = pq.Lattice((4, 4, 4, 8))
>>> bc = pq.BoundaryCondition()
>>> lat.build_neighbor_tables(bc)
>>> U = pq.GaugeField.eye(lat)
>>> psi = pq.SpinorField.random(lat)
>>> params = pq.WilsonParams(mass=0.1)
>>> result = pq.apply_M(U, psi, params, bc)

"""

from dataclasses import dataclass

import torch

from plaq.fields import GaugeField, SpinorField
from plaq.lattice import BoundaryCondition, Lattice


@dataclass
class WilsonParams:
    """Parameters for the Wilson Dirac operator.

    Attributes
    ----------
    mass : float
        Bare mass parameter :math:`m_0`.
    r : float
        Wilson parameter. Fixed to 1.0 for the standard Wilson action.

    Notes
    -----
    The hopping parameter :math:`\\kappa` is related to the mass by:

    .. math::

        \\kappa = \\frac{1}{2(m_0 + 4r)}

    The critical mass (where the pion becomes massless) depends on the
    gauge field configuration, but for free field it is :math:`m_c = 0`.

    Example
    -------
    >>> params = WilsonParams(mass=0.1)
    >>> print(params.kappa)
    0.119...

    """

    mass: float
    r: float = 1.0

    @property
    def kappa(self) -> float:
        """Compute the hopping parameter kappa.

        Returns
        -------
        float
            Hopping parameter :math:`\\kappa = 1/(2(m_0 + 4r))`.

        """
        return 1.0 / (2.0 * (self.mass + 4.0 * self.r))


def apply_M(
    gauge: GaugeField,
    psi: SpinorField,
    params: WilsonParams,
    bc: BoundaryCondition,
) -> SpinorField:
    """Apply the Wilson Dirac operator.

    Computes :math:`M\\psi` where :math:`M` is the Wilson Dirac operator.

    .. math::

        M\\psi(x) = (m_0 + 4r)\\psi(x)
        - \\frac{1}{2}\\sum_{\\mu} \\left[
            (r - \\gamma_\\mu) U_\\mu(x) \\psi(x+\\hat{\\mu})
            + (r + \\gamma_\\mu) U_\\mu^\\dagger(x-\\hat{\\mu}) \\psi(x-\\hat{\\mu})
        \\right]

    Parameters
    ----------
    gauge : GaugeField
        Gauge field configuration.
    psi : SpinorField
        Input spinor field.
    params : WilsonParams
        Wilson operator parameters.
    bc : BoundaryCondition
        Boundary conditions.

    Returns
    -------
    SpinorField
        Result of applying M to psi, in the same layout as input.

    """
    lattice = psi.lattice
    input_layout = psi.layout

    # Ensure neighbor tables are built
    _ensure_neighbor_tables(lattice, bc)

    # Work in site layout
    psi_site = psi.site  # [V, 4, 3]
    result_site = _apply_wilson_site(gauge, psi_site, lattice, params)

    # Create result field in site layout, then convert if needed
    result = SpinorField(result_site, lattice, layout="site")
    return result.as_layout(input_layout)


def apply_Mdag(
    gauge: GaugeField,
    psi: SpinorField,
    params: WilsonParams,
    bc: BoundaryCondition,
) -> SpinorField:
    """Apply the adjoint Wilson Dirac operator.

    Computes :math:`M^\\dagger\\psi` using the :math:`\\gamma_5`-hermiticity relation:

    .. math::

        M^\\dagger = \\gamma_5 M \\gamma_5

    Parameters
    ----------
    gauge : GaugeField
        Gauge field configuration.
    psi : SpinorField
        Input spinor field.
    params : WilsonParams
        Wilson operator parameters.
    bc : BoundaryCondition
        Boundary conditions.

    Returns
    -------
    SpinorField
        Result of applying M^dagger to psi, in the same layout as input.

    """
    lattice = psi.lattice
    input_layout = psi.layout

    # Ensure neighbor tables are built
    _ensure_neighbor_tables(lattice, bc)

    # M^dag = gamma5 M gamma5
    # First apply gamma5
    psi_site = psi.site  # [V, 4, 3]
    gamma5 = _get_gamma5(psi.dtype, psi.device)

    # gamma5 * psi: [4, 4] @ [V, 4, 3] -> need to reshape
    psi_g5 = torch.einsum("ab,vbc->vac", gamma5, psi_site)

    # Apply M
    Mpsi_g5 = _apply_wilson_site(gauge, psi_g5, lattice, params)

    # Apply gamma5 again
    result_site = torch.einsum("ab,vbc->vac", gamma5, Mpsi_g5)

    result = SpinorField(result_site, lattice, layout="site")
    return result.as_layout(input_layout)


def apply_MdagM(
    gauge: GaugeField,
    psi: SpinorField,
    params: WilsonParams,
    bc: BoundaryCondition,
) -> SpinorField:
    """Apply M^dagger M (the normal operator).

    Computes :math:`M^\\dagger M \\psi`. This is a positive semi-definite
    Hermitian operator suitable for use with conjugate gradient solvers.

    Parameters
    ----------
    gauge : GaugeField
        Gauge field configuration.
    psi : SpinorField
        Input spinor field.
    params : WilsonParams
        Wilson operator parameters.
    bc : BoundaryCondition
        Boundary conditions.

    Returns
    -------
    SpinorField
        Result of applying M^dagger M to psi, in the same layout as input.

    """
    Mpsi = apply_M(gauge, psi, params, bc)
    return apply_Mdag(gauge, Mpsi, params, bc)


def _ensure_neighbor_tables(lattice: Lattice, bc: BoundaryCondition) -> None:
    """Ensure neighbor tables are built for the lattice."""
    try:
        lattice.neighbor_fwd(0)
    except RuntimeError:
        lattice.build_neighbor_tables(bc)


def _get_gamma5(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Get gamma5 matrix with specified dtype and device."""
    from plaq.conventions.gamma_milc import get_gamma5

    return get_gamma5(dtype=dtype, device=device)


def _apply_wilson_site(
    gauge: GaugeField,
    psi_site: torch.Tensor,
    lattice: Lattice,
    params: WilsonParams,
) -> torch.Tensor:
    """Apply Wilson operator in site layout.

    This is the core implementation that operates on raw tensors.

    Parameters
    ----------
    gauge : GaugeField
        Gauge field.
    psi_site : torch.Tensor
        Spinor field in site layout [V, 4, 3].
    lattice : Lattice
        Lattice geometry with neighbor tables built.
    params : WilsonParams
        Wilson parameters.

    Returns
    -------
    torch.Tensor
        Result tensor [V, 4, 3].

    """
    dtype = psi_site.dtype
    device = psi_site.device
    m0 = params.mass
    r = params.r

    # Initialize result with diagonal term: (m0 + 4r) * psi
    result = (m0 + 4.0 * r) * psi_site.clone()

    for mu in range(4):
        # Get gauge links for this direction
        U_mu = gauge[mu]  # [V, 3, 3]

        # Forward neighbor: x + hat{mu}
        fwd_idx, fwd_phase = lattice.neighbor_fwd(mu)
        psi_fwd = psi_site[fwd_idx]  # [V, 4, 3]

        # U_mu(x) * psi(x+mu): [V, 3, 3] @ [V, 4, 3] -> [V, 4, 3]
        # Need to contract color indices: U_ab * psi_sb = result_sa
        U_psi_fwd = torch.einsum("vab,vsb->vsa", U_mu, psi_fwd)

        # Apply boundary phase
        U_psi_fwd = U_psi_fwd * fwd_phase.unsqueeze(-1).unsqueeze(-1)

        # Forward term: -1/2 * (r - gamma_mu) * U_mu(x) * psi(x+mu)
        gamma_mu = _get_gamma_mu(mu, dtype, device)
        r_minus_gamma = r * torch.eye(4, dtype=dtype, device=device) - gamma_mu

        # Apply (r - gamma_mu) to U_psi_fwd: [4, 4] @ [V, 4, 3]
        hopping_fwd = torch.einsum("ab,vbc->vac", r_minus_gamma, U_psi_fwd)
        result = result - 0.5 * hopping_fwd

        # Backward neighbor: x - hat{mu}
        bwd_idx, bwd_phase = lattice.neighbor_bwd(mu)
        psi_bwd = psi_site[bwd_idx]  # [V, 4, 3]

        # U_mu^dag(x-mu) * psi(x-mu)
        # U at x-mu points from x-mu to x, so we need U_mu(x-mu)
        U_mu_bwd = gauge[mu][bwd_idx]  # [V, 3, 3]
        U_dag_psi_bwd = torch.einsum("vba,vsb->vsa", U_mu_bwd.conj(), psi_bwd)

        # Apply boundary phase
        U_dag_psi_bwd = U_dag_psi_bwd * bwd_phase.unsqueeze(-1).unsqueeze(-1)

        # Backward term: -1/2 * (r + gamma_mu) * U_mu^dag(x-mu) * psi(x-mu)
        r_plus_gamma = r * torch.eye(4, dtype=dtype, device=device) + gamma_mu

        hopping_bwd = torch.einsum("ab,vbc->vac", r_plus_gamma, U_dag_psi_bwd)
        result = result - 0.5 * hopping_bwd

    return result


def _get_gamma_mu(mu: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Get gamma_mu matrix with specified dtype and device."""
    from plaq.conventions.gamma_milc import gamma

    return gamma[mu].to(dtype=dtype, device=device)
