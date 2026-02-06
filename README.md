# plaq

A lattice gauge theory toolkit for Python, built on PyTorch.

[![CI](https://github.com/ylin910095/plaq/actions/workflows/ci.yml/badge.svg)](https://github.com/ylin910095/plaq/actions/workflows/ci.yml)


> [!WARNING]
> **Disclaimer:** Most codes are AI generated. DO NOT USE IN PRODUCTION.
> Use at your own risk.


## Features

- High-precision computations with `torch.complex128` by default
- Built on PyTorch for GPU acceleration (CPU-only for now)
- Iterative Krylov solvers (CG, BiCGStab) for lattice QCD linear systems
- Even-odd preconditioning for improved solver performance
- Backend abstraction layer for multiple solver implementations (PyTorch, QUDA)
- Type-checked with Pyrefly
- Comprehensive documentation with LaTeX equation support

## Installation

This repository is organized as a **uv workspace** monorepo containing multiple packages:

| Package | Description |
|---------|-------------|
| `plaq` (root) | Core lattice gauge theory toolkit |
| `quda_torch_op` | PyTorch C++ extension for QUDA operators |

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/ylin910095/plaq.git
cd plaq

# Install plaq with all development dependencies (CPU-only, no QUDA)
uv sync --all-groups

# Install plaq with QUDA backend support (requires QUDA_HOME and MPI_HOME env vars)
export QUDA_HOME=/path/to/quda/install  # Path to your QUDA installation
export MPI_HOME=/path/to/mpi            # Path to your MPI installation (e.g., /usr/lib/x86_64-linux-gnu/openmpi)
uv sync --all-groups --extra quda

# Or install all workspace packages
uv sync --all-packages --all-groups
```

### Package-specific Installation

```bash
# Install only the quda_torch_op package
uv sync --package quda_torch_op

# Run tests for a specific package
uv run --package quda_torch_op pytest
```

### Using pip

```bash
# Install plaq only (CPU-only, no QUDA)
pip install -e .

# Install with QUDA support (requires QUDA_HOME and MPI_HOME environment variables)
export QUDA_HOME=/path/to/quda/install  # Path to your QUDA installation
export MPI_HOME=/path/to/mpi            # Path to your MPI installation (e.g., /usr/lib/x86_64-linux-gnu/openmpi)
pip install torch
pip install -e ./packages/quda_torch_op --no-build-isolation
```

## Quickstart

```python
import plaq as pq

# Create a lattice
lat = pq.Lattice((4, 4, 4, 8))
print(f"Volume: {lat.volume}")  # 512

# Create boundary conditions and build neighbor tables
bc = pq.BoundaryCondition()  # Antiperiodic in time, periodic in space
lat.build_neighbor_tables(bc)

# Create fields
U = pq.GaugeField.eye(lat)  # Identity gauge field (free field)
# or
# U = pq.GaugeField.random(lat)  # Haar random SU(3) field
psi = pq.SpinorField.random(lat)  # Random spinor field

# Apply Wilson Dirac operator
params = pq.WilsonParams(mass=0.1)
result = pq.apply_M(U, psi, params, bc)

# Solve linear systems with iterative solvers
b = pq.SpinorField.random(lat)
x, info = pq.solve(U, b, equation="MdagM", params=params, bc=bc)
print(f"Solver converged: {info.converged}, iterations: {info.iters}")

# Access gamma matrices
print(pq.gamma[0])  # gamma_0
print(pq.gamma5)    # gamma_5

# Layout conversion for advanced use
psi_eo = psi.as_layout("eo")  # Convert to even-odd layout
print(psi_eo.eo.shape)  # [2, 256, 4, 3]
```

## Development

### Running Tests

```bash
# Run all tests (root package)
uv run pytest

# Run tests for quda_torch_op package
uv run --package quda_torch_op pytest

# Run all tests including package tests
uv run pytest tests/ packages/quda_torch_op/tests/
```

### Linting and Formatting

```bash
# Check formatting
uv run ruff format --check

# Auto-fix formatting
uv run ruff format

# Run linter
uv run ruff check

# Auto-fix linting issues
uv run ruff check --fix
```

### Type Checking

```bash
uv run pyrefly check
```

### Building Documentation

```bash
uv run sphinx-build -b html docs docs/_build
```

Then open `docs/_build/index.html` in your browser.

### Pre-commit Hooks

The repository uses [pre-commit](https://pre-commit.com/) to enforce formatting, linting, and code hygiene before commits.

**Install the hooks:**

```bash
uv sync --all-groups
uv run pre-commit install
```

**Run on all files:**

```bash
uv run pre-commit run --all-files
```

**Run full checks (including type checking, tests, and docs build):**

```bash
uv run pre-commit run --all-files --hook-stage manual
```

## Project Structure

```
plaq/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI workflow
├── docs/
│   ├── api/                     # API documentation
│   ├── conf.py                  # Sphinx configuration
│   └── index.rst                # Documentation home
├── packages/
│   └── quda_torch_op/           # QUDA PyTorch C++ extension
│       ├── pyproject.toml       # Package configuration
│       ├── setup.py             # C++ extension build script
│       ├── csrc/                # C++ source files
│       ├── quda_torch_op/       # Python package
│       └── tests/               # Package-specific tests
├── src/
│   └── plaq/
│       ├── __init__.py          # Package entry point
│       ├── _version.py          # Version info
│       ├── backends/            # Backend abstraction layer
│       │   ├── __init__.py      # Backend enum, registry, exceptions
│       │   ├── plaq/            # Native plaq backend (CG, BiCGStab)
│       │   └── quda/            # QUDA backend (requires quda_torch_op)
│       ├── config.py            # Global configuration
│       ├── lattice.py           # Lattice and BoundaryCondition
│       ├── layouts.py           # EO packing/unpacking
│       ├── fields.py            # SpinorField, GaugeField
│       ├── conventions/         # Gamma matrices (MILC)
│       ├── operators/           # Wilson Dirac operator
│       ├── solvers/             # High-level solver API
│       ├── precond/             # Preconditioning (even-odd)
│       ├── py.typed             # PEP 561 type marker
│       └── utils/               # Utility functions
├── tests/
│   ├── __init__.py              # Test package init
│   ├── test_backends.py         # Backend abstraction tests
│   ├── test_quda_backend.py     # QUDA backend integration tests
│   ├── test_quda_wilson_comparison.py  # QUDA vs plaq Wilson operator comparison
│   ├── test_gamma.py            # Gamma matrix tests
│   ├── test_layouts.py          # Layout packing tests
│   ├── test_fields.py           # Field tests
│   ├── test_wilson.py           # Wilson operator tests
│   ├── test_solvers.py          # Solver tests (CG, BiCGStab)
│   ├── test_solver_backend.py   # Solver backend tests
│   └── test_placeholder.py      # Basic tests
├── pyproject.toml               # Workspace root configuration
├── uv.lock                      # Workspace lockfile
└── README.md                    # This file
```

## License

MIT License - see LICENSE file for details.
