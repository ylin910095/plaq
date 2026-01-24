"""
quda_torch_op - PyTorch custom operator extension with optional QUDA backend.

This package provides:
- simple_add: A basic tensor addition operator (always available)
- QUDA interface functions (when built with QUDA support)

Usage:
    import quda_torch_op

    # Check if QUDA is available
    if quda_torch_op.quda_is_available():
        quda_torch_op.quda_init(0)  # Initialize on GPU 0
        print(f"QUDA version: {quda_torch_op.quda_get_version()}")
        # ... use QUDA-accelerated functions ...
        quda_torch_op.quda_finalize()  # Clean up

    # simple_add works regardless of QUDA availability
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


def quda_is_available() -> bool:
    """Check if QUDA support is available.

    Returns:
        True if the package was built with QUDA support and QUDA is usable.
    """
    return torch.ops.quda_torch_op.quda_is_available()


def quda_get_device_count() -> int:
    """Get the number of CUDA devices available for QUDA.

    Returns:
        Number of CUDA devices, or 0 if QUDA is not available.
    """
    return torch.ops.quda_torch_op.quda_get_device_count()


def quda_init(device: int = -1) -> None:
    """Initialize QUDA on a specific device.

    This must be called before using any QUDA-accelerated operations.
    The function is idempotent - calling it multiple times with the same
    device is safe.

    Args:
        device: CUDA device index (0-based). Use -1 to select the default device (0).

    Raises:
        RuntimeError: If QUDA is not available, no GPU is found, or if trying
            to reinitialize on a different device.

    Example:
        >>> import quda_torch_op
        >>> if quda_torch_op.quda_is_available():
        ...     quda_torch_op.quda_init(0)  # Initialize on GPU 0
    """
    torch.ops.quda_torch_op.quda_init(device)


def quda_finalize() -> None:
    """Finalize QUDA and release GPU resources.

    This should be called when QUDA is no longer needed to free GPU memory.
    It is safe to call even if QUDA was not initialized.
    """
    torch.ops.quda_torch_op.quda_finalize()


def quda_is_initialized() -> bool:
    """Check if QUDA has been initialized.

    Returns:
        True if quda_init() has been called successfully.
    """
    return torch.ops.quda_torch_op.quda_is_initialized()


def quda_get_device() -> int:
    """Get the device QUDA was initialized on.

    Returns:
        The device index QUDA is using, or -1 if not initialized.
    """
    return torch.ops.quda_torch_op.quda_get_device()


def quda_get_version() -> str:
    """Get the QUDA version string.

    Returns:
        Version string like "1.1.0", or "not available" if QUDA support
        was not compiled in.
    """
    return torch.ops.quda_torch_op.quda_get_version()


__all__ = [
    "__version__",
    "simple_add",
    "quda_is_available",
    "quda_get_device_count",
    "quda_init",
    "quda_finalize",
    "quda_is_initialized",
    "quda_get_device",
    "quda_get_version",
]
