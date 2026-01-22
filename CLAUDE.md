# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**plaq** is a lattice gauge theory toolkit for Python built on PyTorch. It provides data structures and operators for lattice QCD computations with GPU acceleration support.

## Commands

```bash
# Install dependencies
uv sync --all-groups

# Run tests
uv run pytest

# Code quality
uv run ruff format --check    # Check formatting
uv run ruff format            # Auto-fix formatting
uv run ruff check             # Lint
uv run ruff check --fix       # Auto-fix lint issues
uv run pyrefly check          # Type checking

# Pre-commit (runs format + lint)
uv run pre-commit run --all-files

# Full checks including pyrefly, tests, docs
uv run pre-commit run --all-files --hook-stage manual

# Build documentation
uv run sphinx-build -b html docs docs/_build
```

## Architecture

### Core Abstractions (`/src/plaq/`)

- **Lattice** (`lattice.py`): 4D hypercubic lattice geometry with neighbor tables and site indexing
- **SpinorField** / **GaugeField** (`fields.py`): Immutable field containers for fermions (4 spin × 3 color) and SU(3) gauge links
- **BoundaryCondition** (`fields.py`): Fermion boundary conditions (antiperiodic in time, periodic in space)

### Operators and Conventions

- **Wilson Dirac operator** (`operators/wilson.py`): `apply_M`, `apply_Mdag`, `apply_MdagM` implementations
- **Gamma matrices** (`conventions/gamma_milc.py`): MILC convention with `gamma`, `gamma5`, `P_minus`, `P_plus`

### Data Layouts (`layouts.py`)

Two storage layouts with automatic conversion:
- **Site layout**: `[V, 4, 3]` for spinors, `[4, V, 3, 3]` for gauge fields (canonical user-facing)
- **Even-odd layout**: `[2, V/2, 4, 3]` for spinors (optimized for operator applications)

Use `pack_eo` / `unpack_eo` for conversions.

### Backends (`backends/`)

Backend abstraction layer for dispatching solver calls to different implementations:

- **Backend** (`__init__.py`): Enum with `PLAQ` and `QUDA` backends
- **BackendRegistry** (`__init__.py`): Registry for available backend implementations
- **plaq/** (`plaq/`): Native plaq backend package (always available)
  - `cg.py`: Conjugate Gradient for Hermitian positive-definite systems (MdagM)
  - `bicgstab.py`: BiCGStab for general non-Hermitian systems (M)
  - `__init__.py`: High-level plaq solver and backend registration

```python
# Check backend availability
from plaq.backends import Backend, registry
registry.is_available(Backend.PLAQ)  # True
registry.is_available(Backend.QUDA)  # False (requires separate installation)
```

### Solvers (`solvers/`)

High-level solver API for lattice QCD linear systems:

- **solve** (`api.py`): High-level API with auto-selection of solver and equation type
- **SolverInfo** (`api.py`): Dataclass for solver convergence information

The low-level Krylov solvers (`cg`, `bicgstab`) are implemented in `backends/plaq/`
and re-exported from `solvers/` for convenience.

```python
# Example usage
x, info = pq.solve(U, b, equation="MdagM")  # Uses CG
x, info = pq.solve(U, b, equation="M")      # Uses BiCGStab
```

### Preconditioning (`precond/`)

- **Even-odd** (`even_odd.py`): Schur complement preconditioning for MdagM equation

### Global Configuration (`config.py`)

`plaq.config` singleton controls dtype (default: `torch.complex128`) and device (default: `cpu`).

## Conventions

- Immutable field objects: use `.clone()` for modification
- NumPy-style docstrings with LaTeX math notation
- All code is Pyrefly type-checked
- Line length: 100 characters
