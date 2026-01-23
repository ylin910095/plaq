"""
Tests for the quda_torch_op.simple_add operator.
"""

import pytest
import torch


class TestSimpleAddRegistration:
    """Tests to verify the operator is properly registered."""

    def test_namespace_exists(self):
        """Verify quda_torch_op namespace is registered."""
        import quda_torch_op  # noqa: F401

        assert hasattr(torch.ops, "quda_torch_op"), (
            "Namespace 'quda_torch_op' not found in torch.ops"
        )

    def test_operator_exists(self):
        """Verify simple_add operator is registered in namespace."""
        import quda_torch_op  # noqa: F401

        assert hasattr(torch.ops.quda_torch_op, "simple_add"), (
            "Operator 'simple_add' not found in torch.ops.quda_torch_op"
        )


class TestSimpleAddCPU:
    """Tests for CPU functionality of simple_add."""

    def test_basic_addition(self):
        """Test basic element-wise addition on CPU."""
        import quda_torch_op  # noqa: F401

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])

        out = torch.ops.quda_torch_op.simple_add(a, b)

        assert torch.allclose(out, a + b), "Output does not match expected a + b"
        assert out.device.type == "cpu", f"Expected CPU tensor, got {out.device}"

    def test_random_tensors(self):
        """Test addition with random tensors."""
        import quda_torch_op  # noqa: F401

        a = torch.randn(100, 50)
        b = torch.randn(100, 50)

        out = torch.ops.quda_torch_op.simple_add(a, b)

        assert torch.allclose(out, a + b), "Output does not match expected a + b"
        assert out.device.type == "cpu", f"Expected CPU tensor, got {out.device}"

    def test_different_dtypes(self):
        """Test addition preserves dtype."""
        import quda_torch_op  # noqa: F401

        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
            a = torch.ones(10, dtype=dtype)
            b = torch.ones(10, dtype=dtype)

            out = torch.ops.quda_torch_op.simple_add(a, b)

            assert out.dtype == dtype, f"Expected dtype {dtype}, got {out.dtype}"
            assert torch.allclose(out, a + b), f"Output mismatch for dtype {dtype}"

    def test_multidimensional(self):
        """Test addition with multi-dimensional tensors."""
        import quda_torch_op  # noqa: F401

        a = torch.randn(2, 3, 4, 5)
        b = torch.randn(2, 3, 4, 5)

        out = torch.ops.quda_torch_op.simple_add(a, b)

        assert torch.allclose(out, a + b), "Output does not match for 4D tensors"
        assert out.shape == a.shape, f"Shape mismatch: {out.shape} vs {a.shape}"

    def test_noncontiguous_input(self):
        """Test that non-contiguous tensors are handled correctly."""
        import quda_torch_op  # noqa: F401

        # Create non-contiguous tensors via transpose
        a = torch.randn(10, 20).t()  # 20x10, non-contiguous
        b = torch.randn(10, 20).t()  # 20x10, non-contiguous

        assert not a.is_contiguous(), "Test setup error: a should be non-contiguous"
        assert not b.is_contiguous(), "Test setup error: b should be non-contiguous"

        out = torch.ops.quda_torch_op.simple_add(a, b)

        assert torch.allclose(out, a + b), "Output mismatch for non-contiguous input"


class TestSimpleAddErrors:
    """Tests for error handling in simple_add."""

    def test_shape_mismatch_raises_error(self):
        """Test that mismatched shapes raise a clear error."""
        import quda_torch_op  # noqa: F401

        a = torch.randn(3, 4)
        b = torch.randn(4, 3)

        with pytest.raises(RuntimeError, match="same shape"):
            torch.ops.quda_torch_op.simple_add(a, b)

    def test_dtype_mismatch_raises_error(self):
        """Test that mismatched dtypes raise a clear error."""
        import quda_torch_op  # noqa: F401

        a = torch.randn(10, dtype=torch.float32)
        b = torch.randn(10, dtype=torch.float64)

        with pytest.raises(RuntimeError, match="same dtype"):
            torch.ops.quda_torch_op.simple_add(a, b)
