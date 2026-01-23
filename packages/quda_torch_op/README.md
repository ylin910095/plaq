# quda_torch_op

A minimal PyTorch custom operator extension for QUDA-backed operators.

This repository provides a clean skeleton for building PyTorch C++ extensions using the stable ABI. It includes a simple `simple_add` operator as a demonstration.

## Features

- Uses PyTorch's stable C++ API (`STABLE_TORCH_LIBRARY`)
- CPU-only implementation (CUDA/QUDA ready structure)
- Proper operator registration via `torch.ops` namespace
- Requires Python 3.11+ and PyTorch 2.6+

## Installation

### 1. Create environment with uv

```bash
uv venv --python 3.11
source .venv/bin/activate
```

### 2. Install PyTorch

```bash
uv pip install torch
```

### 3. Build and install in editable mode

```bash
uv pip install -e . --no-build-isolation -v
```

## Running Tests

```bash
# Install test dependencies
uv pip install pytest

# Run tests
pytest -q tests/
```

## Usage

```python
import torch
import quda_torch_op

# Create tensors
a = torch.randn(3, 4)
b = torch.randn(3, 4)

# Call the operator directly
result = quda_torch_op.simple_add(a, b)

# Or via torch.ops namespace
result = torch.ops.quda_torch_op.simple_add(a, b)

# Verify result
assert torch.allclose(result, a + b)
print("simple_add works correctly!")
```

## Project Structure

```
quda_torch_op/
├── pyproject.toml          # Build system configuration
├── setup.py                # Extension build script
├── README.md
├── quda_torch_op/
│   ├── __init__.py         # Package init, loads C++ extension
│   └── _version.py         # Version string
├── csrc/
│   └── code.cpp            # CPU implementation + op registration
└── tests/
    └── test_simple_add.py  # pytest tests
```

## API Reference

### `quda_torch_op.simple_add(a, b) -> Tensor`

Element-wise addition of two tensors.

**Arguments:**
- `a` (Tensor): First input tensor (CPU)
- `b` (Tensor): Second input tensor (CPU, same shape and dtype as `a`)

**Returns:**
- Tensor: Element-wise sum `a + b`

**Raises:**
- `RuntimeError`: If tensors have different shapes, dtypes, or devices

## Development

To extend this skeleton with QUDA-backed operators:

1. Add CUDA sources to `csrc/` directory
2. Update `setup.py` to use `CUDAExtension` when CUDA is available
3. Register CUDA implementations with `STABLE_TORCH_LIBRARY_IMPL(quda_torch_op, CUDA, m)`

## License

MIT
