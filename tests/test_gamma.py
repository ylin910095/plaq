"""Tests for MILC gamma matrices.

This module tests the Clifford algebra relations and gamma5 properties.
"""

import torch

import plaq as pq


def test_gamma_clifford_algebra() -> None:
    """Test that gamma matrices satisfy the Clifford algebra.

    Verifies: {gamma_mu, gamma_nu} = 2 delta_{mu,nu} I
    """
    dtype = torch.complex128
    device = torch.device("cpu")
    identity = torch.eye(4, dtype=dtype, device=device)
    tol = 1e-14

    for mu in range(4):
        for nu in range(4):
            gamma_mu = pq.gamma[mu].to(dtype=dtype, device=device)
            gamma_nu = pq.gamma[nu].to(dtype=dtype, device=device)

            # Anticommutator: {gamma_mu, gamma_nu} = gamma_mu @ gamma_nu + gamma_nu @ gamma_mu
            anticommutator = gamma_mu @ gamma_nu + gamma_nu @ gamma_mu

            # Expected: 2 * delta_{mu,nu} * I
            expected = 2.0 * identity if mu == nu else torch.zeros(4, 4, dtype=dtype, device=device)

            diff = torch.abs(anticommutator - expected).max().item()
            assert diff < tol, f"Clifford algebra violation for mu={mu}, nu={nu}: max diff = {diff}"


def test_gamma5_product() -> None:
    """Test that gamma5 = gamma_0 @ gamma_1 @ gamma_2 @ gamma_3."""
    dtype = torch.complex128
    device = torch.device("cpu")
    tol = 1e-14

    gamma5_computed = (
        pq.gamma[0].to(dtype=dtype, device=device)
        @ pq.gamma[1].to(dtype=dtype, device=device)
        @ pq.gamma[2].to(dtype=dtype, device=device)
        @ pq.gamma[3].to(dtype=dtype, device=device)
    )
    gamma5_expected = pq.gamma5.to(dtype=dtype, device=device)

    diff = torch.abs(gamma5_computed - gamma5_expected).max().item()
    assert diff < tol, f"gamma5 product mismatch: max diff = {diff}"


def test_gamma5_squared() -> None:
    """Test that gamma5^2 = I."""
    dtype = torch.complex128
    device = torch.device("cpu")
    identity = torch.eye(4, dtype=dtype, device=device)
    tol = 1e-14

    gamma5 = pq.gamma5.to(dtype=dtype, device=device)
    gamma5_sq = gamma5 @ gamma5

    diff = torch.abs(gamma5_sq - identity).max().item()
    assert diff < tol, f"gamma5^2 != I: max diff = {diff}"


def test_gamma5_anticommutes_with_gamma() -> None:
    """Test that {gamma5, gamma_mu} = 0 for all mu."""
    dtype = torch.complex128
    device = torch.device("cpu")
    tol = 1e-14

    gamma5 = pq.gamma5.to(dtype=dtype, device=device)

    for mu in range(4):
        gamma_mu = pq.gamma[mu].to(dtype=dtype, device=device)
        anticommutator = gamma5 @ gamma_mu + gamma_mu @ gamma5

        norm = torch.abs(anticommutator).max().item()
        assert norm < tol, f"gamma5 does not anticommute with gamma_{mu}: {norm}"


def test_projectors_idempotent() -> None:
    """Test that P_plus^2 = P_plus and P_minus^2 = P_minus."""
    dtype = torch.complex128
    device = torch.device("cpu")
    tol = 1e-14

    for mu in range(4):
        p_plus = pq.P_plus(mu, dtype=dtype, device=device)
        p_minus = pq.P_minus(mu, dtype=dtype, device=device)

        # P^2 = P
        diff_plus = torch.abs(p_plus @ p_plus - p_plus).max().item()
        diff_minus = torch.abs(p_minus @ p_minus - p_minus).max().item()

        assert diff_plus < tol, f"P_plus({mu}) not idempotent: {diff_plus}"
        assert diff_minus < tol, f"P_minus({mu}) not idempotent: {diff_minus}"


def test_projectors_sum_to_identity() -> None:
    """Test that P_plus + P_minus = I."""
    dtype = torch.complex128
    device = torch.device("cpu")
    identity = torch.eye(4, dtype=dtype, device=device)
    tol = 1e-14

    for mu in range(4):
        p_plus = pq.P_plus(mu, dtype=dtype, device=device)
        p_minus = pq.P_minus(mu, dtype=dtype, device=device)

        diff = torch.abs(p_plus + p_minus - identity).max().item()
        assert diff < tol, f"P_plus({mu}) + P_minus({mu}) != I: {diff}"


def test_projectors_orthogonal() -> None:
    """Test that P_plus @ P_minus = 0."""
    dtype = torch.complex128
    device = torch.device("cpu")
    tol = 1e-14

    for mu in range(4):
        p_plus = pq.P_plus(mu, dtype=dtype, device=device)
        p_minus = pq.P_minus(mu, dtype=dtype, device=device)

        product = p_plus @ p_minus
        norm = torch.abs(product).max().item()

        assert norm < tol, f"P_plus({mu}) @ P_minus({mu}) != 0: {norm}"
