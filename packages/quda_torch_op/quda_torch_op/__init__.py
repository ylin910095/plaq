"""
quda_torch_op - A minimal PyTorch custom operator extension.

Usage:
    import quda_torch_op
    result = quda_torch_op.simple_add(a, b)
"""

import torch

# Import the C++ extension to trigger operator registration
import quda_torch_op._C  # noqa: F401
from quda_torch_op._version import __version__


def simple_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise addition of two tensors.

    Args:
        a: First input tensor (CPU)
        b: Second input tensor (CPU, same shape and dtype as a)

    Returns:
        Element-wise sum a + b
    """
    return torch.ops.quda_torch_op.simple_add(a, b)


__all__ = ["__version__", "simple_add"]
