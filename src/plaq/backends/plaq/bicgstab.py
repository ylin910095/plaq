"""BiCGStab solver for general non-Hermitian systems.

This module implements the Bi-Conjugate Gradient Stabilized (BiCGStab) method
for solving linear systems :math:`A x = b` with a general (non-Hermitian) matrix.

Algorithm
---------
The Van der Vorst BiCGStab algorithm:

.. math::

    r_0 &= b - A x_0 \\\\
    \\\\hat{r}_0 &= r_0 \\\\quad \\\\text{(shadow residual, fixed)} \\\\\\\\
    \\\\rho_0 &= 1, \\\\quad \\\\alpha = 1, \\\\quad \\\\omega_0 = 1 \\\\\\\\
    v_0 &= p_0 = 0 \\\\\\\\
    \\\\text{for } k &= 1, 2, \\ldots \\\\\\\\
    \\\\rho_k &= (\\\\hat{r}_0, r_{k-1}) \\\\\\\\
    \\\\beta &= (\\\\rho_k / \\\\rho_{k-1}) \\\\cdot (\\\\alpha / \\\\omega_{k-1}) \\\\\\\\
    p_k &= r_{k-1} + \\\\beta (p_{k-1} - \\\\omega_{k-1} v_{k-1}) \\\\\\\\
    v_k &= A p_k \\\\\\\\
    \\\\alpha &= \\\\rho_k / (\\\\hat{r}_0, v_k) \\\\\\\\
    s &= r_{k-1} - \\\\alpha v_k \\\\\\\\
    t &= A s \\\\\\\\
    \\\\omega_k &= (t, s) / (t, t) \\\\\\\\
    x_k &= x_{k-1} + \\\\alpha p_k + \\\\omega_k s \\\\\\\\
    r_k &= s - \\\\omega_k t

BiCGStab combines BiCG with GMRES(1) stabilization to avoid the erratic
convergence behavior of pure BiCG.

Notes
-----
For lattice QCD, BiCGStab is used to solve :math:`M x = b` directly where
:math:`M` is the Wilson Dirac operator. This is useful when the source
is not a standard Hermitian system.

Example
-------
>>> import torch
>>> def A_apply(x):
...     # Example: shifted diagonal matrix
...     return 2.0 * x + 0.1j * x
>>> b = torch.randn(100, dtype=torch.complex128)
>>> x, info = bicgstab(A_apply, b, tol=1e-10)
>>> print(f"Converged: {info.converged}, iters: {info.iters}")

"""

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class BiCGStabInfo:
    """Information about BiCGStab solver convergence.

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


def bicgstab(
    A_apply: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    tol: float = 1e-10,
    maxiter: int = 1000,
    callback: Callable[[torch.Tensor], None] | None = None,
) -> tuple[torch.Tensor, BiCGStabInfo]:
    """BiCGStab solver for general non-Hermitian systems.

    Solves :math:`A x = b` where :math:`A` is a general (possibly non-Hermitian)
    matrix.

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
    tuple[torch.Tensor, BiCGStabInfo]
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
        return b.clone(), BiCGStabInfo(converged=True, iters=0, final_residual=0.0)

    # Initialize
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = x0.clone()
        Ax = A_apply(x)
        r = b - Ax

    # Shadow residual (fixed throughout)
    r_hat = r.clone()

    # Initial scalars
    rho = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    alpha = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    omega = torch.tensor(1.0, dtype=b.dtype, device=b.device)

    # Initial vectors
    v = torch.zeros_like(b)
    p = torch.zeros_like(b)

    r_flat = r.flatten()
    final_residual = torch.linalg.norm(r_flat).real.item() / b_norm
    converged = False
    iters = 0

    for _k in range(maxiter):
        iters = _k + 1
        # Check convergence
        if final_residual < tol:
            converged = True
            break

        # rho_new = (r_hat, r)
        r_hat_flat = r_hat.flatten()
        r_flat = r.flatten()
        rho_new = torch.vdot(r_hat_flat, r_flat)

        # Check for breakdown
        if rho_new.abs() < 1e-30:
            break

        # beta = (rho_new / rho) * (alpha / omega)
        beta = (rho_new / rho) * (alpha / omega)

        # p = r + beta * (p - omega * v)
        p = r + beta * (p - omega * v)

        # v = A @ p
        v = A_apply(p)
        v_flat = v.flatten()

        # alpha = rho_new / (r_hat, v)
        r_hat_v = torch.vdot(r_hat_flat, v_flat)
        if r_hat_v.abs() < 1e-30:
            break
        alpha = rho_new / r_hat_v

        # s = r - alpha * v
        s = r - alpha * v
        s_flat = s.flatten()

        # Check if s is small enough (early termination)
        s_norm = torch.linalg.norm(s_flat).real.item()
        if s_norm / b_norm < tol:
            x = x + alpha * p
            final_residual = s_norm / b_norm
            converged = True
            # Call user callback with current solution
            if callback is not None:
                callback(x)
            break

        # t = A @ s
        t = A_apply(s)
        t_flat = t.flatten()

        # omega = (t, s) / (t, t)
        t_s = torch.vdot(t_flat, s_flat)
        t_t = torch.vdot(t_flat, t_flat)
        if t_t.abs() < 1e-30:
            break
        omega = t_s / t_t

        # x = x + alpha * p + omega * s
        x = x + alpha * p + omega * s

        # Call user callback with current solution
        if callback is not None:
            callback(x)

        # r = s - omega * t
        r = s - omega * t
        r_flat = r.flatten()

        # Update residual and rho
        final_residual = torch.linalg.norm(r_flat).real.item() / b_norm
        rho = rho_new

    return x, BiCGStabInfo(converged=converged, iters=iters, final_residual=final_residual)
