"""
Tests for quda_torch_op QUDA interface.

These tests verify that the QUDA interface functions work correctly.
Tests are split into:
- Tests that always run (QUDA availability check)
- Tests that require QUDA (skipped if QUDA not available)

Note: QUDA finalization may fail in some container environments due to
NVML initialization issues. Tests are designed to work around this.
"""

import pytest
import quda_torch_op


class TestQudaAvailability:
    """Tests for QUDA availability check (always run)."""

    def test_quda_is_available_returns_bool(self):
        """quda_is_available should return a boolean."""
        result = quda_torch_op.quda_is_available()
        assert isinstance(result, bool)

    def test_quda_get_version_returns_string(self):
        """quda_get_version should return a string."""
        result = quda_torch_op.quda_get_version()
        assert isinstance(result, str)
        # Should be a version string or "not available"
        assert result == "not available" or "." in result

    def test_quda_get_device_count_returns_int(self):
        """quda_get_device_count should return a non-negative integer."""
        result = quda_torch_op.quda_get_device_count()
        assert isinstance(result, int)
        assert result >= 0


@pytest.mark.skipif(
    not quda_torch_op.quda_is_available(),
    reason="QUDA not available",
)
class TestQudaInterface:
    """Tests for QUDA interface (require QUDA to be available)."""

    def test_quda_get_device_count_positive(self):
        """When QUDA is available, device count should be positive."""
        assert quda_torch_op.quda_get_device_count() > 0

    def test_quda_version_format(self):
        """QUDA version should have proper format."""
        version = quda_torch_op.quda_get_version()
        # Should look like "X.Y.Z"
        parts = version.split(".")
        assert len(parts) >= 2
        # Major and minor should be numeric
        assert parts[0].isdigit()
        assert parts[1].isdigit()

    def test_quda_init(self):
        """Test QUDA initialization.

        Note: We don't test finalization here because QUDA's endQuda()
        may fail in container environments due to NVML issues.
        The process cleanup will handle resource release.
        """
        # Initialize QUDA on device 0
        quda_torch_op.quda_init(0)
        assert quda_torch_op.quda_is_initialized()
        assert quda_torch_op.quda_get_device() == 0

        # Test idempotent init (same device)
        quda_torch_op.quda_init(0)  # Should not raise
        quda_torch_op.quda_init(-1)  # -1 means default, should also be OK
        assert quda_torch_op.quda_is_initialized()

        # Note: Not testing finalization due to NVML issues in containers


@pytest.mark.skipif(
    quda_torch_op.quda_is_available(),
    reason="Test for when QUDA is not available",
)
class TestQudaNotAvailable:
    """Tests for when QUDA is not available."""

    def test_quda_init_raises_when_not_available(self):
        """quda_init should raise RuntimeError when QUDA not available."""
        with pytest.raises(RuntimeError, match="QUDA is not available"):
            quda_torch_op.quda_init(0)

    def test_quda_version_not_available(self):
        """quda_get_version should return 'not available'."""
        assert quda_torch_op.quda_get_version() == "not available"

    def test_quda_device_count_zero(self):
        """quda_get_device_count should return 0."""
        assert quda_torch_op.quda_get_device_count() == 0
