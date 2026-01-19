# plaq

A lattice gauge theory toolkit for Python, built on PyTorch.

[![CI](https://github.com/YOUR_USERNAME/plaq/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/plaq/actions/workflows/ci.yml)

## Features

- High-precision computations with `torch.complex128` by default
- Built on PyTorch for GPU acceleration (CPU-only for now)
- Type-checked with Pyre
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

# Check the version
print(pq.__version__)  # 0.1.0

# Check default configuration
print(pq.config.DEFAULT_DTYPE)  # torch.complex128
print(pq.config.DEFAULT_DEVICE)  # cpu

# Modify configuration if needed
import torch
pq.config.DEFAULT_DTYPE = torch.complex64

# Reset to defaults
pq.config.reset()
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
uv run pyre check
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
│       └── ci.yml          # GitHub Actions CI workflow
├── docs/
│   ├── api/                # API documentation
│   ├── conf.py             # Sphinx configuration
│   └── index.rst           # Documentation home
├── src/
│   └── plaq/
│       ├── __init__.py     # Package entry point
│       ├── _version.py     # Version info
│       ├── config.py       # Global configuration
│       ├── py.typed        # PEP 561 type marker
│       └── utils/          # Utility functions
├── tests/
│   └── test_placeholder.py # Placeholder tests
├── .pyre_configuration     # Pyre type checker config
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## License

MIT License - see LICENSE file for details.
