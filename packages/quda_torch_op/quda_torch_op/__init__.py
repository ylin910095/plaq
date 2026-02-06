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


def quda_wilson_mat(
    gauge: torch.Tensor,
    psi: torch.Tensor,
    kappa: float,
    antiperiodic_t: bool = True,
) -> torch.Tensor:
    """Apply the Wilson Dirac operator M using QUDA.

    Computes M * psi where M is the Wilson-Dirac operator with the specified
    hopping parameter kappa.

    Args:
        gauge: Gauge field tensor with shape [4, V, 3, 3] in plaq layout.
            The first index is the direction (x, y, z, t), V is the lattice
            volume, and the last two indices are color (SU(3) matrix).
        psi: Spinor field tensor with shape [V, 4, 3] in plaq layout.
            The indices are (site, spin, color).
        kappa: Hopping parameter. Related to bare mass by kappa = 1/(2*(m0 + 4r)).
        antiperiodic_t: If True, use antiperiodic boundary conditions in the
            time direction (standard for fermions). If False, use periodic BC.

    Returns:
        Result spinor tensor with shape [V, 4, 3].

    Raises:
        RuntimeError: If QUDA is not initialized or not available.

    Example:
        >>> import quda_torch_op
        >>> quda_torch_op.quda_init(0)
        >>> # gauge has shape [4, V, 3, 3], psi has shape [V, 4, 3]
        >>> result = quda_torch_op.quda_wilson_mat(gauge, psi, kappa=0.125)
        >>> quda_torch_op.quda_finalize()
    """
    return torch.ops.quda_torch_op.quda_wilson_mat(gauge, psi, kappa, antiperiodic_t)


def quda_wilson_mat_dag(
    gauge: torch.Tensor,
    psi: torch.Tensor,
    kappa: float,
    antiperiodic_t: bool = True,
) -> torch.Tensor:
    """Apply the adjoint Wilson Dirac operator M^dag using QUDA.

    Computes M^dag * psi where M^dag is the Hermitian conjugate of the
    Wilson-Dirac operator.

    Args:
        gauge: Gauge field tensor with shape [4, V, 3, 3] in plaq layout.
        psi: Spinor field tensor with shape [V, 4, 3] in plaq layout.
        kappa: Hopping parameter. Related to bare mass by kappa = 1/(2*(m0 + 4r)).
        antiperiodic_t: If True, use antiperiodic boundary conditions in the
            time direction. If False, use periodic BC.

    Returns:
        Result spinor tensor with shape [V, 4, 3].

    Raises:
        RuntimeError: If QUDA is not initialized or not available.

    Note:
        For the Wilson operator, M^dag = gamma5 * M * gamma5 (gamma5-hermiticity).
    """
    return torch.ops.quda_torch_op.quda_wilson_mat_dag(gauge, psi, kappa, antiperiodic_t)


def quda_wilson_mat_dag_mat(
    gauge: torch.Tensor,
    psi: torch.Tensor,
    kappa: float,
    antiperiodic_t: bool = True,
) -> torch.Tensor:
    """Apply M^dag * M (the normal operator) using QUDA.

    Computes M^dag * M * psi, which is a Hermitian positive semi-definite
    operator suitable for conjugate gradient solvers.

    Args:
        gauge: Gauge field tensor with shape [4, V, 3, 3] in plaq layout.
        psi: Spinor field tensor with shape [V, 4, 3] in plaq layout.
        kappa: Hopping parameter. Related to bare mass by kappa = 1/(2*(m0 + 4r)).
        antiperiodic_t: If True, use antiperiodic boundary conditions in the
            time direction. If False, use periodic BC.

    Returns:
        Result spinor tensor with shape [V, 4, 3].

    Raises:
        RuntimeError: If QUDA is not initialized or not available.

    Note:
        M^dag * M is Hermitian and positive semi-definite, making it suitable
        for solving linear systems with the conjugate gradient method.
    """
    return torch.ops.quda_torch_op.quda_wilson_mat_dag_mat(gauge, psi, kappa, antiperiodic_t)


__all__ = [
    "__version__",
    "quda_finalize",
    "quda_get_device",
    "quda_get_device_count",
    "quda_get_version",
    "quda_init",
    "quda_is_available",
    "quda_is_initialized",
    "quda_wilson_mat",
    "quda_wilson_mat_dag",
    "quda_wilson_mat_dag_mat",
    "simple_add",
]
