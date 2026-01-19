"""Placeholder tests for plaq.

This module contains placeholder tests to verify the testing infrastructure works.
"""

import torch

import plaq as pq


def test_version_exists() -> None:
    """Test that version string is defined."""
    assert pq.__version__ is not None
    assert isinstance(pq.__version__, str)
    assert len(pq.__version__) > 0


def test_config_default_dtype() -> None:
    """Test that default dtype is complex128."""
    assert torch.complex128 == pq.config.DEFAULT_DTYPE


def test_config_default_device() -> None:
    """Test that default device is CPU."""
    assert torch.device("cpu") == pq.config.DEFAULT_DEVICE


def test_config_reset() -> None:
    """Test that config reset works correctly."""
    # Change config
    pq.config.DEFAULT_DTYPE = torch.complex64

    # Reset
    pq.config.reset()

    # Verify reset
    assert torch.complex128 == pq.config.DEFAULT_DTYPE
