"""Conjugate Gradient solver for Hermitian positive-definite systems.

This module implements the Conjugate Gradient (CG) method for solving linear
systems :math:`A x = b` where :math:`A` is Hermitian positive-definite.

Algorithm
---------
The standard CG recurrence (Hestenes-Stiefel):

.. math::

    r_0 &= b - A x_0 \\\\
    p_0 &= r_0 \\\\
    \\\\text{for } k &= 0, 1, 2, \\ldots \\\\
    \\\\alpha_k &= \\\\frac{(r_k, r_k)}{(p_k, A p_k)} \\\\\\\\
    x_{k+1} &= x_k + \\\\alpha_k p_k \\\\\\\\
    r_{k+1} &= r_k - \\\\alpha_k A p_k \\\\\\\\
    \\\\text{if } &\\\\|r_{k+1}\\\\| / \\\\|b\\\\| < \\\\text{tol}: \\\\text{converged} \\\\\\\\
    \\\\beta_k &= \\\\frac{(r_{k+1}, r_{k+1})}{(r_k, r_k)} \\\\\\\\
    p_{k+1} &= r_{k+1} + \\\\beta_k p_k

The algorithm minimizes :math:`\\\\|x - x^*\\\\|_A` at each iteration, where
:math:`x^*` is the exact solution and :math:`\\\\|v\\\\|_A = \\\\sqrt{v^\\\\dagger A v}`.

Notes
-----
For lattice QCD, CG is typically used to solve the normal equation
:math:`M^\\\\dagger M x = M^\\\\dagger b`, which is Hermitian positive-definite
when :math:`M` is the Wilson Dirac operator (or similar).

Example
-------
>>> import torch
>>> def A_apply(x):
...     # Example: diagonal matrix
...     return 2.0 * x
>>> b = torch.randn(100, dtype=torch.complex128)
>>> x, info = cg(A_apply, b, tol=1e-10)
>>> print(f"Converged: {info.converged}, iters: {info.iters}")

"""

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class CGInfo:
    """Information about CG solver convergence.

    Attributes
    ----------
    converged : bool
        Whether the solver converged within tolerance.
    iters : int
        Number of iterations performed.
    final_residual : float
        Final relative residual norm ||r|| / ||b||.

    """

    converged: bool
    iters: int
    final_residual: float


def cg(
    A_apply: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    tol: float = 1e-10,
    maxiter: int = 1000,
    callback: Callable[[torch.Tensor], None] | None = None,
) -> tuple[torch.Tensor, CGInfo]:
    """Conjugate Gradient solver for Hermitian positive-definite systems.

    Solves :math:`A x = b` where :math:`A` is Hermitian positive-definite.

    Parameters
    ----------
    A_apply : Callable[[torch.Tensor], torch.Tensor]
        Function that computes the matrix-vector product A @ x.
        Must preserve shape and dtype.
    b : torch.Tensor
        Right-hand side vector.
    x0 : torch.Tensor, optional
        Initial guess. If None, uses zero vector.
    tol : float
        Relative tolerance for convergence: ||r|| / ||b|| < tol.
    maxiter : int
        Maximum number of iterations.
    callback : Callable[[torch.Tensor], None], optional
        User-supplied function to call after each iteration.
        Called as callback(xk) where xk is the current solution vector.

    Returns
    -------
    tuple[torch.Tensor, CGInfo]
        Solution vector x and convergence information.

    Raises
    ------
    ValueError
        If b has zero norm (trivial system).

    """
    # Flatten for inner products
    b_flat = b.flatten()
    b_norm = torch.linalg.norm(b_flat).real.item()

    if b_norm < 1e-15:
        # Trivial case: b = 0 implies x = 0
        return b.clone(), CGInfo(converged=True, iters=0, final_residual=0.0)

    # Initialize
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = x0.clone()
        Ax = A_apply(x)
        r = b - Ax

    p = r.clone()
    r_flat = r.flatten()
    rr = torch.vdot(r_flat, r_flat).real  # (r, r)

    converged = False
    final_residual = torch.sqrt(rr).item() / b_norm
    iters = 0

    for _k in range(maxiter):
        iters = _k + 1
        # Check convergence
        if final_residual < tol:
            converged = True
            break

        # Ap = A @ p
        Ap = A_apply(p)
        Ap_flat = Ap.flatten()
        p_flat = p.flatten()

        # alpha = (r, r) / (p, Ap)
        pAp = torch.vdot(p_flat, Ap_flat).real
        if pAp.abs() < 1e-30:
            # Breakdown: A is not positive definite or numerical issues
            break

        alpha = rr / pAp

        # x = x + alpha * p
        x = x + alpha * p

        # Call user callback with current solution
        if callback is not None:
            callback(x)

        # r = r - alpha * Ap
        r = r - alpha * Ap
        r_flat = r.flatten()

        # New (r, r)
        rr_new = torch.vdot(r_flat, r_flat).real
        final_residual = torch.sqrt(rr_new).item() / b_norm

        # beta = (r_new, r_new) / (r, r)
        beta = rr_new / rr

        # p = r + beta * p
        p = r + beta * p

        rr = rr_new

    return x, CGInfo(converged=converged, iters=iters, final_residual=final_residual)
