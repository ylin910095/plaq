"""Even-odd (Schur complement) preconditioning for Wilson operator.

This module implements even-odd preconditioning which exploits the
checkerboard structure of the Wilson operator to reduce the linear
system size by a factor of 2.

Block Structure
---------------
The Wilson operator in even-odd block form:

.. math::

    M = \\begin{pmatrix} M_{ee} & M_{eo} \\\\ M_{oe} & M_{oo} \\end{pmatrix}

where:
- :math:`M_{ee}, M_{oo}` are the diagonal blocks (on-site mass terms)
- :math:`M_{eo}, M_{oe}` are off-diagonal blocks (hopping terms)

For the Wilson operator, the diagonal blocks are simply:

.. math::

    M_{ee} = M_{oo} = (m_0 + 4r) \\cdot I

since the hopping terms only connect even to odd sites and vice versa.

Schur Complement
----------------
The Schur complement on even sites is:

.. math::

    \\hat{M}_{ee} = M_{ee} - M_{eo} M_{oo}^{-1} M_{oe}

The preconditioned system solves on the even sites only, then reconstructs
the odd sites:

.. math::

    \\hat{M}_{ee} x_e &= b_e - M_{eo} M_{oo}^{-1} b_o \\\\
    x_o &= M_{oo}^{-1} (b_o - M_{oe} x_e)

For MdagM, we apply the Schur complement to :math:`M^\\dagger M`.

Notes
-----
The preconditioning reduces the condition number and halves the system
size, both of which speed up convergence.

"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from plaq.backends.plaq.cg import CGInfo, cg
from plaq.fields import GaugeField, SpinorField
from plaq.lattice import BoundaryCondition
from plaq.layouts import pack_eo, unpack_eo
from plaq.operators import WilsonParams

if TYPE_CHECKING:
    from plaq.lattice import Lattice


def _get_mass_factor(params: WilsonParams) -> float:
    """Get the diagonal mass factor (m0 + 4r)."""
    return params.mass + 4.0 * params.r


def apply_Mee_inv(
    psi_e: torch.Tensor,
    params: WilsonParams,
) -> torch.Tensor:
    """Apply inverse of M_ee (diagonal block on even sites).

    For Wilson: M_ee = (m0 + 4r) * I, so M_ee^{-1} = 1/(m0 + 4r) * I.

    Parameters
    ----------
    psi_e : torch.Tensor
        Even-site spinor data with shape [V/2, 4, 3].
    params : WilsonParams
        Wilson operator parameters.

    Returns
    -------
    torch.Tensor
        Result of applying M_ee^{-1}.

    """
    mass_factor = _get_mass_factor(params)
    return psi_e / mass_factor


def apply_Moo_inv(
    psi_o: torch.Tensor,
    params: WilsonParams,
) -> torch.Tensor:
    """Apply inverse of M_oo (diagonal block on odd sites).

    For Wilson: M_oo = (m0 + 4r) * I, so M_oo^{-1} = 1/(m0 + 4r) * I.

    Parameters
    ----------
    psi_o : torch.Tensor
        Odd-site spinor data with shape [V/2, 4, 3].
    params : WilsonParams
        Wilson operator parameters.

    Returns
    -------
    torch.Tensor
        Result of applying M_oo^{-1}.

    """
    mass_factor = _get_mass_factor(params)
    return psi_o / mass_factor


def apply_hopping_eo(
    gauge: GaugeField,
    psi_o: torch.Tensor,
    lattice: "Lattice",
    params: WilsonParams,
) -> torch.Tensor:
    """Apply M_eo hopping term (odd to even sites).

    Computes the hopping contribution from odd sites to even sites:

    .. math::

        (M_{eo} \\psi_o)_e = -\\frac{1}{2}\\sum_\\mu \\left[
            (r - \\gamma_\\mu) U_\\mu(x_e) \\psi_o(x_e + \\hat{\\mu})
            + (r + \\gamma_\\mu) U_\\mu^\\dagger(x_e - \\hat{\\mu}) \\psi_o(x_e - \\hat{\\mu})
        \\right]

    Note: Forward neighbor of even site is odd, backward neighbor of even is odd.

    Parameters
    ----------
    gauge : GaugeField
        Gauge field configuration.
    psi_o : torch.Tensor
        Odd-site spinor data with shape [V/2, 4, 3].
    lattice : Lattice
        Lattice geometry.
    params : WilsonParams
        Wilson operator parameters.

    Returns
    -------
    torch.Tensor
        Result on even sites with shape [V/2, 4, 3].

    """
    return _apply_hopping_block(gauge, psi_o, lattice, params, src_parity=1, _dst_parity=0)


def apply_hopping_oe(
    gauge: GaugeField,
    psi_e: torch.Tensor,
    lattice: "Lattice",
    params: WilsonParams,
) -> torch.Tensor:
    """Apply M_oe hopping term (even to odd sites).

    Parameters
    ----------
    gauge : GaugeField
        Gauge field configuration.
    psi_e : torch.Tensor
        Even-site spinor data with shape [V/2, 4, 3].
    lattice : Lattice
        Lattice geometry.
    params : WilsonParams
        Wilson operator parameters.

    Returns
    -------
    torch.Tensor
        Result on odd sites with shape [V/2, 4, 3].

    """
    return _apply_hopping_block(gauge, psi_e, lattice, params, src_parity=0, _dst_parity=1)


def _apply_hopping_block(
    gauge: GaugeField,
    psi_src: torch.Tensor,
    lattice: "Lattice",
    params: WilsonParams,
    src_parity: int,
    _dst_parity: int,
) -> torch.Tensor:
    """Apply hopping term between parity sectors.

    This is the core hopping implementation that connects one parity
    sector to the other.

    """
    from plaq.conventions.gamma_milc import gamma

    dtype = psi_src.dtype
    device = psi_src.device
    r = params.r

    # Get site index mappings
    even_sites = lattice.even_sites
    odd_sites = lattice.odd_sites

    if src_parity == 0:
        src_sites = even_sites
        dst_sites = odd_sites
    else:
        src_sites = odd_sites
        dst_sites = even_sites

    # Create full-volume tensor for source
    psi_full = torch.zeros(lattice.volume, 4, 3, dtype=dtype, device=device)
    psi_full[src_sites] = psi_src

    # Result tensor
    result_full = torch.zeros(lattice.volume, 4, 3, dtype=dtype, device=device)

    for mu in range(4):
        U_mu = gauge[mu]  # [V, 3, 3]
        gamma_mu = gamma[mu].to(dtype=dtype, device=device)

        # Forward: dst_sites get contribution from src_sites (their forward neighbors)
        fwd_idx, fwd_phase = lattice.neighbor_fwd(mu)
        psi_fwd = psi_full[fwd_idx]  # [V, 4, 3]

        # U_mu(x) @ psi(x + mu)
        U_psi_fwd = torch.einsum("vab,vsb->vsa", U_mu, psi_fwd)
        U_psi_fwd = U_psi_fwd * fwd_phase.unsqueeze(-1).unsqueeze(-1)

        # (r - gamma_mu) @ U_psi_fwd
        r_minus_gamma = r * torch.eye(4, dtype=dtype, device=device) - gamma_mu
        hopping_fwd = torch.einsum("ab,vbc->vac", r_minus_gamma, U_psi_fwd)
        result_full = result_full - 0.5 * hopping_fwd

        # Backward: dst_sites get contribution from src_sites (their backward neighbors)
        bwd_idx, bwd_phase = lattice.neighbor_bwd(mu)
        psi_bwd = psi_full[bwd_idx]  # [V, 4, 3]

        # U_mu^dag(x - mu) @ psi(x - mu)
        U_mu_bwd = gauge[mu][bwd_idx]  # [V, 3, 3]
        U_dag_psi_bwd = torch.einsum("vba,vsb->vsa", U_mu_bwd.conj(), psi_bwd)
        U_dag_psi_bwd = U_dag_psi_bwd * bwd_phase.unsqueeze(-1).unsqueeze(-1)

        # (r + gamma_mu) @ U_dag_psi_bwd
        r_plus_gamma = r * torch.eye(4, dtype=dtype, device=device) + gamma_mu
        hopping_bwd = torch.einsum("ab,vbc->vac", r_plus_gamma, U_dag_psi_bwd)
        result_full = result_full - 0.5 * hopping_bwd

    # Extract destination parity
    return result_full[dst_sites]


def apply_schur_complement(
    gauge: GaugeField,
    psi_e: torch.Tensor,
    lattice: "Lattice",
    params: WilsonParams,
) -> torch.Tensor:
    """Apply Schur complement operator on even sites.

    Computes:

    .. math::

        \\hat{M}_{ee} \\psi_e = M_{ee} \\psi_e - M_{eo} M_{oo}^{-1} M_{oe} \\psi_e

    Parameters
    ----------
    gauge : GaugeField
        Gauge field.
    psi_e : torch.Tensor
        Even-site spinor [V/2, 4, 3].
    lattice : Lattice
        Lattice geometry.
    params : WilsonParams
        Wilson parameters.

    Returns
    -------
    torch.Tensor
        Result on even sites [V/2, 4, 3].

    """
    mass_factor = _get_mass_factor(params)

    # M_ee psi_e
    Mee_psi = mass_factor * psi_e

    # M_oe psi_e (even -> odd)
    Moe_psi = apply_hopping_oe(gauge, psi_e, lattice, params)

    # M_oo^{-1} M_oe psi_e
    Moo_inv_Moe_psi = apply_Moo_inv(Moe_psi, params)

    # M_eo M_oo^{-1} M_oe psi_e (odd -> even)
    Meo_Moo_inv_Moe_psi = apply_hopping_eo(gauge, Moo_inv_Moe_psi, lattice, params)

    # Schur complement: M_ee - M_eo M_oo^{-1} M_oe
    return Mee_psi - Meo_Moo_inv_Moe_psi


def apply_schur_complement_dag(
    gauge: GaugeField,
    psi_e: torch.Tensor,
    lattice: "Lattice",
    params: WilsonParams,
) -> torch.Tensor:
    """Apply adjoint Schur complement on even sites.

    Uses gamma5-hermiticity: M^dag = gamma5 M gamma5.

    """
    from plaq.conventions.gamma_milc import get_gamma5

    dtype = psi_e.dtype
    device = psi_e.device
    gamma5 = get_gamma5(dtype=dtype, device=device)

    # gamma5 psi_e
    g5_psi = torch.einsum("ab,vbc->vac", gamma5, psi_e)

    # M_schur gamma5 psi_e
    M_g5_psi = apply_schur_complement(gauge, g5_psi, lattice, params)

    # gamma5 M_schur gamma5 psi_e = M_schur^dag psi_e
    return torch.einsum("ab,vbc->vac", gamma5, M_g5_psi)


def apply_schur_MdagM(
    gauge: GaugeField,
    psi_e: torch.Tensor,
    lattice: "Lattice",
    params: WilsonParams,
) -> torch.Tensor:
    """Apply M_schur^dag M_schur on even sites.

    This is the preconditioned normal operator for CG.

    """
    M_psi = apply_schur_complement(gauge, psi_e, lattice, params)
    return apply_schur_complement_dag(gauge, M_psi, lattice, params)


def solve_eo_preconditioned(
    gauge: GaugeField,
    b_data: torch.Tensor,
    lattice: "Lattice",
    params: WilsonParams,
    bc: BoundaryCondition,
    tol: float,
    maxiter: int,
    dtype: torch.dtype,
    callback: Callable[[torch.Tensor], None] | None = None,
) -> tuple[torch.Tensor, CGInfo]:
    """Solve MdagM system with even-odd preconditioning.

    Solves :math:`M^\\dagger M x = M^\\dagger b` using proper block structure
    of MdagM with iterative solves.

    The key insight is that the block structure of MdagM is:

    .. math::

        (M^\\dagger M)_{ee} = M^\\dagger_{ee} M_{ee} + M^\\dagger_{eo} M_{oe}
        (M^\\dagger M)_{oo} = M^\\dagger_{oe} M_{eo} + M^\\dagger_{oo} M_{oo}

    These are NOT diagonal! They involve hopping terms.

    Algorithm:
    1. Compute c = M^dag b
    2. Define (MdagM)_ee and (MdagM)_oo block operators via full MdagM application
    3. Solve (MdagM)_ee x_e = c_e - (MdagM)_eo (MdagM)_oo^{-1} c_o (Schur complement)
    4. Reconstruct x_o = (MdagM)_oo^{-1} (c_o - (MdagM)_oe x_e)

    Since (MdagM)_oo is not diagonal, we use CG to invert it.

    Parameters
    ----------
    gauge : GaugeField
        Gauge field.
    b_data : torch.Tensor
        RHS in site layout [V, 4, 3].
    lattice : Lattice
        Lattice geometry.
    params : WilsonParams
        Wilson parameters.
    bc : BoundaryCondition
        Boundary conditions.
    tol : float
        Solver tolerance.
    maxiter : int
        Maximum iterations.
    dtype : torch.dtype
        Working dtype.
    callback : Callable[[torch.Tensor], None], optional
        User-supplied function to call after each iteration of the outer CG solve.
        Called as callback(xk) where xk is the current solution in site layout.

    Returns
    -------
    tuple[torch.Tensor, CGInfo]
        Solution in site layout and solver info.

    """
    # Ensure neighbor tables built
    try:
        lattice.neighbor_fwd(0)
    except RuntimeError:
        lattice.build_neighbor_tables(bc)

    from plaq.operators import apply_Mdag, apply_MdagM

    # Step 1: Compute c = M^dag b (the RHS for MdagM solve)
    b_field = SpinorField(b_data, lattice, layout="site")
    c_field = apply_Mdag(gauge, b_field, params, bc)
    c = c_field.site.to(dtype=dtype)

    # Step 2: Pack c into EO layout
    c_eo = pack_eo(c, lattice)  # [2, V/2, 4, 3]
    c_e = c_eo[0]  # Even sites
    c_o = c_eo[1]  # Odd sites

    # Define block operators for MdagM via full operator application
    def apply_MdagM_ee(x_e: torch.Tensor) -> torch.Tensor:
        """Apply (MdagM)_ee: input even, output even."""
        x_eo = torch.stack([x_e, torch.zeros_like(x_e)], dim=0)
        x_full = unpack_eo(x_eo, lattice)
        x_field = SpinorField(x_full, lattice, layout="site")
        y = apply_MdagM(gauge, x_field, params, bc)
        y_eo = pack_eo(y.site, lattice)
        return y_eo[0]  # Even part

    def apply_MdagM_oo(x_o: torch.Tensor) -> torch.Tensor:
        """Apply (MdagM)_oo: input odd, output odd."""
        x_eo = torch.stack([torch.zeros_like(x_o), x_o], dim=0)
        x_full = unpack_eo(x_eo, lattice)
        x_field = SpinorField(x_full, lattice, layout="site")
        y = apply_MdagM(gauge, x_field, params, bc)
        y_eo = pack_eo(y.site, lattice)
        return y_eo[1]  # Odd part

    def apply_MdagM_eo(x_o: torch.Tensor) -> torch.Tensor:
        """Apply (MdagM)_eo: input odd, output even."""
        x_eo = torch.stack([torch.zeros_like(x_o), x_o], dim=0)
        x_full = unpack_eo(x_eo, lattice)
        x_field = SpinorField(x_full, lattice, layout="site")
        y = apply_MdagM(gauge, x_field, params, bc)
        y_eo = pack_eo(y.site, lattice)
        return y_eo[0]  # Even part

    def apply_MdagM_oe(x_e: torch.Tensor) -> torch.Tensor:
        """Apply (MdagM)_oe: input even, output odd."""
        x_eo = torch.stack([x_e, torch.zeros_like(x_e)], dim=0)
        x_full = unpack_eo(x_eo, lattice)
        x_field = SpinorField(x_full, lattice, layout="site")
        y = apply_MdagM(gauge, x_field, params, bc)
        y_eo = pack_eo(y.site, lattice)
        return y_eo[1]  # Odd part

    # Step 3: Solve (MdagM)_oo^{-1} c_o using CG
    # This is needed for the Schur complement RHS
    Moo_inv_co, info_prec = cg(apply_MdagM_oo, c_o, tol=tol, maxiter=maxiter)

    # Compute (MdagM)_eo (MdagM)_oo^{-1} c_o
    Meo_Moo_inv_co = apply_MdagM_eo(Moo_inv_co)

    # Preconditioned RHS: c_e - (MdagM)_eo (MdagM)_oo^{-1} c_o
    rhs_e = c_e - Meo_Moo_inv_co

    # Step 4: Define Schur complement operator and solve
    # Schur = (MdagM)_ee - (MdagM)_eo (MdagM)_oo^{-1} (MdagM)_oe
    def apply_schur_MdagM_full(x_e: torch.Tensor) -> torch.Tensor:
        """Apply the Schur complement of MdagM on even sites."""
        # (MdagM)_ee x_e
        Mee_x = apply_MdagM_ee(x_e)

        # (MdagM)_oe x_e
        Moe_x = apply_MdagM_oe(x_e)

        # (MdagM)_oo^{-1} (MdagM)_oe x_e using CG
        Moo_inv_Moe_x, _ = cg(apply_MdagM_oo, Moe_x, tol=tol * 0.1, maxiter=maxiter)

        # (MdagM)_eo (MdagM)_oo^{-1} (MdagM)_oe x_e
        Meo_Moo_inv_Moe_x = apply_MdagM_eo(Moo_inv_Moe_x)

        # Schur complement: (MdagM)_ee - (MdagM)_eo (MdagM)_oo^{-1} (MdagM)_oe
        return Mee_x - Meo_Moo_inv_Moe_x

    # Wrap callback to convert even-site layout to full site layout
    even_callback = None
    if callback is not None:
        callback_fn = callback  # Capture in local scope for type checker

        def even_callback(x_e_iter: torch.Tensor) -> None:
            # Reconstruct temporary full solution for callback
            # Note: odd sites are zero during this solve
            x_eo_temp = torch.stack([x_e_iter, torch.zeros_like(x_e_iter)], dim=0)
            x_full_temp = unpack_eo(x_eo_temp, lattice)
            callback_fn(x_full_temp)

    # Solve Schur x_e = rhs_e using CG
    x_e, info_even = cg(
        apply_schur_MdagM_full, rhs_e, tol=tol, maxiter=maxiter, callback=even_callback
    )

    # Step 5: Reconstruct x_o
    # x_o = (MdagM)_oo^{-1} (c_o - (MdagM)_oe x_e)
    Moe_xe = apply_MdagM_oe(x_e)
    rhs_o = c_o - Moe_xe
    x_o, info_odd = cg(apply_MdagM_oo, rhs_o, tol=tol, maxiter=maxiter)

    # Pack back to full volume
    x_eo = torch.stack([x_e, x_o], dim=0)  # [2, V/2, 4, 3]
    x_full = unpack_eo(x_eo, lattice)  # [V, 4, 3]

    # Return combined info
    total_iters = info_prec.iters + info_even.iters + info_odd.iters
    converged = info_prec.converged and info_even.converged and info_odd.converged

    # Compute final residual of the full system
    x_field = SpinorField(x_full, lattice, layout="site")
    MdagM_x = apply_MdagM(gauge, x_field, params, bc)
    residual = torch.linalg.norm((MdagM_x.site - c).flatten())
    c_norm = torch.linalg.norm(c.flatten())
    final_residual = (residual / c_norm).real.item()

    return x_full, CGInfo(converged=converged, iters=total_iters, final_residual=final_residual)
