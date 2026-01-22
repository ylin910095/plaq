# plaq

A lattice gauge theory toolkit for Python, built on PyTorch.

[![CI](https://github.com/ylin910095/plaq/actions/workflows/ci.yml/badge.svg)](https://github.com/ylin910095/plaq/actions/workflows/ci.yml)

## Features

- High-precision computations with `torch.complex128` by default
- Built on PyTorch for GPU acceleration (CPU-only for now)
- Iterative Krylov solvers (CG, BiCGStab) for lattice QCD linear systems
- Even-odd preconditioning for improved solver performance
- Backend abstraction layer for multiple solver implementations (PyTorch, QUDA)
- Type-checked with Pyrefly
- Comprehensive documentation with LaTeX equation support

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/plaq.git
cd plaq

# Install with all development dependencies
uv sync --all-groups
```

### Using pip

```bash
pip install -e .
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
x, info = pq.solve(U, b, equation="MdagM", mass=0.1, boundary_condition=bc)
print(f"Solver converged: {info['converged']}, iterations: {info['iterations']}")

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
uv run pytest
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
│       └── ci.yml              # GitHub Actions CI workflow
├── docs/
│   ├── api/                    # API documentation
│   ├── conf.py                 # Sphinx configuration
│   └── index.rst               # Documentation home
├── src/
│   └── plaq/
│       ├── __init__.py         # Package entry point
│       ├── _version.py         # Version info
│       ├── backends/           # Backend abstraction layer
│       │   ├── __init__.py     # Backend enum, registry, exceptions
│       │   └── plaq/            # Native plaq backend (CG, BiCGStab)
│       ├── config.py           # Global configuration
│       ├── lattice.py          # Lattice and BoundaryCondition
│       ├── layouts.py          # EO packing/unpacking
│       ├── fields.py           # SpinorField, GaugeField
│       ├── conventions/        # Gamma matrices (MILC)
│       ├── operators/          # Wilson Dirac operator
│       ├── solvers/            # High-level solver API
│       ├── precond/            # Preconditioning (even-odd)
│       ├── py.typed            # PEP 561 type marker
│       └── utils/              # Utility functions
├── tests/
│   ├── test_backends.py        # Backend abstraction tests
│   ├── test_gamma.py           # Gamma matrix tests
│   ├── test_layouts.py         # Layout packing tests
│   ├── test_fields.py          # Field tests
│   ├── test_wilson.py          # Wilson operator tests
│   ├── test_solvers.py         # Solver tests (CG, BiCGStab)
│   └── test_placeholder.py     # Basic tests
├── pyproject.toml              # Project configuration (includes Pyrefly config)
└── README.md                   # This file
```

## License

MIT License - see LICENSE file for details.
