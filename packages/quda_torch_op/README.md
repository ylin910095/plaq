# quda_torch_op

A PyTorch custom operator extension with optional QUDA backend support for GPU-accelerated lattice QCD computations.

## Features

- Uses PyTorch's stable C++ API (`STABLE_TORCH_LIBRARY`)
- Optional QUDA backend for GPU-accelerated operations
- Proper operator registration via `torch.ops` namespace
- Requires Python 3.11+ and PyTorch 2.6+

## Requirements

### System Dependencies

For QUDA support, you need:

1. **CUDA Toolkit** (runtime libraries are sufficient)
2. **QUDA library** compiled with MPI support
3. **OpenMPI** development headers

Install OpenMPI development headers on Ubuntu/Debian:

```bash
apt-get install libopenmpi-dev
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `QUDA_HOME` | Path to QUDA installation directory | `/opt/quda/install` |

## Installation

### From the plaq monorepo (recommended)

The easiest way to install is from the main plaq project:

```bash
cd /path/to/plaq

# Install with QUDA support
QUDA_HOME=/opt/quda/install uv sync --all-packages --all-groups

# Or install without QUDA (CPU-only)
uv sync --all-packages --all-groups
```

### Standalone installation

```bash
# With QUDA support
QUDA_HOME=/opt/quda/install pip install .

# Without QUDA (CPU-only)
pip install .
```

## Build Output

During build, you'll see one of these messages:

- `QUDA status: QUDA found at /path/to/quda with N GPU(s) available.` - QUDA support enabled
- `QUDA status: QUDA_HOME environment variable not set...` - Building without QUDA
- `QUDA status: No CUDA-capable GPU found...` - Building without QUDA

## Usage

### Basic Usage (always available)

```python
import torch
import quda_torch_op

# Simple tensor addition (CPU)
a = torch.randn(3, 4)
b = torch.randn(3, 4)
result = quda_torch_op.simple_add(a, b)
```

### QUDA Interface

```python
import quda_torch_op

# Check if QUDA is available
if quda_torch_op.quda_is_available():
    print(f"QUDA version: {quda_torch_op.quda_get_version()}")
    print(f"GPU count: {quda_torch_op.quda_get_device_count()}")

    # Initialize QUDA on GPU 0
    quda_torch_op.quda_init(0)

    # ... use QUDA-accelerated operations ...

    # Clean up (optional - resources freed on process exit)
    quda_torch_op.quda_finalize()
else:
    print("QUDA not available - using CPU fallback")
```

## API Reference

### Core Operations

#### `simple_add(a, b) -> Tensor`

Element-wise addition of two tensors.

**Arguments:**
- `a` (Tensor): First input tensor (CPU)
- `b` (Tensor): Second input tensor (CPU, same shape and dtype as `a`)

**Returns:** Element-wise sum `a + b`

### QUDA Interface

#### `quda_is_available() -> bool`

Check if QUDA support is available.

#### `quda_get_device_count() -> int`

Get the number of CUDA devices available for QUDA.

#### `quda_get_version() -> str`

Get the QUDA version string (e.g., "1.1.0").

#### `quda_init(device: int = -1) -> None`

Initialize QUDA on a specific device.

**Arguments:**
- `device`: CUDA device index (0-based). Use -1 for default device (0).

**Note:** This function is idempotent - calling multiple times with the same device is safe.

#### `quda_finalize() -> None`

Finalize QUDA and release GPU resources.

#### `quda_is_initialized() -> bool`

Check if QUDA has been initialized.

#### `quda_get_device() -> int`

Get the device QUDA was initialized on (-1 if not initialized).

## Troubleshooting

### QUDA not found

If you see "QUDA_HOME environment variable not set", make sure:

1. QUDA is installed (e.g., at `/opt/quda/install`)
2. Set the environment variable: `export QUDA_HOME=/opt/quda/install`
3. Verify QUDA installation contains `lib/libquda.so` and `include/quda.h`

### No GPU found

If you see "No CUDA-capable GPU found", verify:

1. NVIDIA drivers are installed: `nvidia-smi`
2. CUDA is available to PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`

### MPI errors

QUDA requires MPI. If you see MPI-related errors:

1. Install OpenMPI dev headers: `apt-get install libopenmpi-dev`
2. Ensure MPI libraries are in the library path

## Project Structure

```
quda_torch_op/
├── pyproject.toml           # Build system configuration
├── setup.py                 # Extension build script with QUDA detection
├── README.md
├── quda_torch_op/
│   ├── __init__.py          # Package init, QUDA interface
│   └── _version.py          # Version string
├── csrc/
│   ├── code.cpp             # Base operators + stubs when QUDA unavailable
│   └── quda_interface.cpp   # QUDA implementation (when QUDA_ENABLED)
└── tests/
    ├── test_simple_add.py   # CPU operator tests
    └── test_quda_interface.py  # QUDA interface tests
```

## License

MIT
