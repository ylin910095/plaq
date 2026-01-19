"""Tests for field containers and SU(3) sampling.

This module tests:
- GaugeField.random produces valid SU(3) matrices (unitary, det=1)
- SU(3) matrices follow the Haar measure distribution
"""

import torch

import plaq as pq


class TestGaugeFieldRandom:
    """Tests for GaugeField.random SU(3) sampling."""

    def test_random_gauge_is_unitary(self) -> None:
        """Random gauge matrices should satisfy U^dag U = I."""
        lat = pq.Lattice((4, 4, 4, 8))
        torch.manual_seed(12345)
        U = pq.GaugeField.random(lat)

        # Check U^dag U = I for all matrices
        data = U.data  # [4, V, 3, 3]
        U_dag = data.conj().transpose(-2, -1)  # [4, V, 3, 3]
        product = torch.einsum("mvab,mvbc->mvac", U_dag, data)  # [4, V, 3, 3]

        # Should be close to identity
        identity = torch.eye(3, dtype=data.dtype, device=data.device)
        identity = identity.unsqueeze(0).unsqueeze(0).expand(4, lat.volume, 3, 3)

        error = torch.abs(product - identity).max().item()
        assert error < 1e-12, f"U^dag U differs from identity by {error}"

    def test_random_gauge_has_unit_determinant(self) -> None:
        """Random gauge matrices should have determinant 1."""
        lat = pq.Lattice((4, 4, 4, 8))
        torch.manual_seed(54321)
        U = pq.GaugeField.random(lat)

        # Compute determinant of all matrices
        data = U.data  # [4, V, 3, 3]
        det = torch.linalg.det(data)  # [4, V]

        # Determinant should be 1 (unit complex number with phase 0)
        det_abs = torch.abs(det)
        det_phase = torch.angle(det)

        abs_error = torch.abs(det_abs - 1.0).max().item()
        phase_error = torch.abs(det_phase).max().item()

        assert abs_error < 1e-12, f"|det| differs from 1 by {abs_error}"
        assert phase_error < 1e-12, f"arg(det) differs from 0 by {phase_error}"

    def test_random_gauge_is_special_unitary(self) -> None:
        """Combined test: U in SU(3) means U^dag U = I and det(U) = 1."""
        lat = pq.Lattice((2, 2, 2, 4))
        torch.manual_seed(99999)
        U = pq.GaugeField.random(lat)

        data = U.data

        # Unitarity: U^dag U = I
        U_dag = data.conj().transpose(-2, -1)
        product = torch.einsum("mvab,mvbc->mvac", U_dag, data)
        identity = torch.eye(3, dtype=data.dtype, device=data.device)
        unitarity_error = torch.abs(product - identity).max().item()

        # Special: det(U) = 1
        det = torch.linalg.det(data)
        det_error = torch.abs(det - 1.0).max().item()

        assert unitarity_error < 1e-12, f"Unitarity error: {unitarity_error}"
        assert det_error < 1e-12, f"Determinant error: {det_error}"


class TestHaarMeasureDistribution:
    """Tests for proper Haar measure distribution of SU(3) matrices."""

    def test_trace_distribution_is_correct(self) -> None:
        """The trace of Haar-distributed SU(3) should have known statistics.

        For SU(N), E[Tr(U)] = 0 and E[|Tr(U)|^2] = 1.
        This is a fundamental property of the Haar measure.
        """
        lat = pq.Lattice((8, 8, 8, 16))  # Large sample for statistics
        torch.manual_seed(77777)
        U = pq.GaugeField.random(lat)

        data = U.data  # [4, V, 3, 3]
        traces = torch.einsum("mvaa->mv", data)  # [4, V] complex

        # Mean trace should be close to 0
        mean_trace = traces.mean()
        mean_trace_abs = torch.abs(mean_trace).item()

        # Mean |trace|^2 should be close to 1
        trace_sq = torch.abs(traces) ** 2
        mean_trace_sq = trace_sq.mean().item()

        # With 4 * 8^3 * 16 = 131072 samples, statistical error is small
        n_samples = 4 * lat.volume
        expected_std_mean = 1.0 / n_samples**0.5  # ~0.003

        assert mean_trace_abs < 10 * expected_std_mean, (
            f"E[Tr(U)] = {mean_trace_abs:.4f}, expected ~0 (within {10 * expected_std_mean:.4f})"
        )
        assert abs(mean_trace_sq - 1.0) < 0.1, f"E[|Tr(U)|^2] = {mean_trace_sq:.4f}, expected 1.0"

    def test_matrix_elements_have_correct_variance(self) -> None:
        """Individual matrix elements should have variance 1/N for SU(N).

        For uniformly distributed SU(N), each matrix element U_{ij} satisfies:
        - E[U_{ij}] = 0
        - E[|U_{ij}|^2] = 1/N = 1/3 for SU(3)
        """
        lat = pq.Lattice((8, 8, 8, 16))
        torch.manual_seed(88888)
        U = pq.GaugeField.random(lat)

        data = U.data  # [4, V, 3, 3]

        # Flatten to get all matrix elements
        elements = data.reshape(-1)  # [4 * V * 9] complex

        # Mean should be ~0
        mean_element = elements.mean()
        mean_abs = torch.abs(mean_element).item()

        # Mean |element|^2 should be ~1/3
        element_sq = torch.abs(elements) ** 2
        mean_element_sq = element_sq.mean().item()

        assert mean_abs < 0.01, f"E[U_ij] = {mean_abs:.4f}, expected ~0"
        assert abs(mean_element_sq - 1.0 / 3.0) < 0.02, (
            f"E[|U_ij|^2] = {mean_element_sq:.4f}, expected {1.0 / 3.0:.4f}"
        )

    def test_different_seeds_give_different_results(self) -> None:
        """Verify that different seeds produce different gauge fields."""
        lat = pq.Lattice((2, 2, 2, 2))

        torch.manual_seed(111)
        U1 = pq.GaugeField.random(lat)

        torch.manual_seed(222)
        U2 = pq.GaugeField.random(lat)

        # Should be different
        diff = torch.abs(U1.data - U2.data).max().item()
        assert diff > 0.1, "Different seeds should produce different results"

    def test_same_seed_gives_same_result(self) -> None:
        """Verify reproducibility with same seed."""
        lat = pq.Lattice((2, 2, 2, 2))

        torch.manual_seed(333)
        U1 = pq.GaugeField.random(lat)

        torch.manual_seed(333)
        U2 = pq.GaugeField.random(lat)

        # Should be identical
        diff = torch.abs(U1.data - U2.data).max().item()
        assert diff < 1e-15, f"Same seed should give same result, diff={diff}"
