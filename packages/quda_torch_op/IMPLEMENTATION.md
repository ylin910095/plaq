# QUDA Wilson Solver Implementation Details

This document describes the implementation details for interfacing the plaq lattice QCD library with QUDA's Wilson Dslash solver.

## Table of Contents

1. [Requirements](#requirements)
2. [Overview](#overview)
3. [Gauge Field Storage Format](#gauge-field-storage-format)
4. [Spinor Field Storage Format](#spinor-field-storage-format)
5. [Even-Odd (Checkerboard) Ordering](#even-odd-checkerboard-ordering)
6. [Gamma Matrix Convention](#gamma-matrix-convention)
7. [Wilson Dirac Operator](#wilson-dirac-operator)
8. [Solver Configuration](#solver-configuration)
9. [Boundary Conditions](#boundary-conditions)
10. [Data Copy Notes](#data-copy-notes)

---

## Requirements

### QUDA Build Requirements

QUDA must be built with at least one gauge field interface enabled. The QDP interface is recommended as it is enabled by default and has the simplest conversion from plaq's format.

**CMake options (required):**
```bash
cmake \
  -DQUDA_INTERFACE_QDP=ON \           # Required (default: ON) - this interface must be enabled
  -DQUDA_INTERFACE_MILC=ON \          # Alternative (default: ON)
  -DQUDA_DIRAC_WILSON=ON \            # Required (default: ON)
  -DQUDA_GPU_ARCH=sm_XX \             # Set to your GPU architecture
  ...
```

**IMPORTANT:** If you see errors like:
```
ERROR: QDP interface has not been built
 (rank 0, host ..., copy_gauge_inc.cu:204 ...)
```

Your QUDA installation was compiled without the QDP interface. Rebuild QUDA with `-DQUDA_INTERFACE_QDP=ON` explicitly specified.

**Full QUDA build example:**
```bash
cd /path/to/quda
mkdir build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/opt/quda/install \
  -DQUDA_INTERFACE_QDP=ON \
  -DQUDA_INTERFACE_MILC=ON \
  -DQUDA_DIRAC_WILSON=ON \
  -DQUDA_GPU_ARCH=sm_86 \
  -DQUDA_MPI=ON
make -j$(nproc)
make install
```

### Environment Variables

```bash
export QUDA_HOME=/path/to/quda/install
export MPI_HOME=/path/to/mpi  # Required even for single-GPU builds
```

---

## Overview

The QUDA backend enables GPU-accelerated Wilson fermion inversions using the QUDA library. The implementation handles conversion between plaq's native storage format and QUDA's expected formats.

### Key Components

- **C++ Interface** (`csrc/wilson_interface.cpp`): Handles data conversion and QUDA API calls
- **Python Interface** (`quda_torch_op/__init__.py`): Exposes `wilson_invert()` function
- **plaq Backend** (`src/plaq/backends/quda/__init__.py`): Integrates with plaq's solver API

---

## Gauge Field Storage Format

### plaq Format

Gauge fields in plaq use **site layout** with shape `[4, V, 3, 3]`:

$$U_{\mu}^{ab}(x) \to \text{gauge}[\mu, \text{site}(x), a, b]$$

Where:
- First index: direction $\mu \in \{0, 1, 2, 3\}$ corresponding to $(x, y, z, t)$
- Second index: site in **lexicographic order**
- Last two indices: color matrix entries (row $a$, column $b$) for $a, b \in \{0, 1, 2\}$

**Lexicographic site ordering:**
$$\text{site}(x, y, z, t) = x + N_x \cdot (y + N_y \cdot (z + N_z \cdot t))$$

where $x$ varies fastest.

### QUDA QDP Format

QUDA with `QUDA_QDP_GAUGE_ORDER` expects:
- Array of 4 pointers: `gauge[mu]` for $\mu \in \{0, 1, 2, 3\}$
- Each `gauge[mu]` points to `[V, 3, 3]` complex data
- Sites are in **even-odd ordering** (even sites first, then odd sites)
- Color matrix is stored row-major: `[row, col]`

### Conversion Algorithm

```
for each direction mu:
    for each site in lexicographic order:
        eo_site = lex_to_eo[lex_site]
        gauge_quda[mu][eo_site, :, :] = gauge_plaq[mu, lex_site, :, :]
```

---

## Spinor Field Storage Format

### plaq Format

Spinor fields in plaq use **site layout** with shape `[V, 4, 3]`:

$$\psi^{\alpha a}(x) \to \text{spinor}[\text{site}(x), \alpha, a]$$

Where:
- First index: site in **lexicographic order**
- Second index: spin index $\alpha \in \{0, 1, 2, 3\}$
- Third index: color index $a \in \{0, 1, 2\}$

### QUDA Format

QUDA with `QUDA_DIRAC_ORDER` expects:
- Shape `[V, 4, 3]` (same as plaq)
- Sites in **even-odd ordering**
- Spin-color ordering: color inside spin (matches plaq)

### Conversion Algorithm

```
for each site in lexicographic order:
    eo_site = lex_to_eo[lex_site]
    spinor_quda[eo_site, :, :] = spinor_plaq[lex_site, :, :]
```

---

## Even-Odd (Checkerboard) Ordering

The even-odd decomposition is based on site parity:

$$\text{parity}(x, y, z, t) = (x + y + z + t) \mod 2$$

- **Even sites**: parity = 0
- **Odd sites**: parity = 1

### Even-Odd to Lexicographic Map

The even-odd index maps to lexicographic index as follows:

```
eo_to_lex = []
# First, enumerate all even sites
for t in range(Nt):
    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):
                if (x + y + z + t) % 2 == 0:
                    eo_to_lex.append(x + Nx*(y + Ny*(z + Nz*t)))

# Then, enumerate all odd sites
for t in range(Nt):
    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):
                if (x + y + z + t) % 2 == 1:
                    eo_to_lex.append(x + Nx*(y + Ny*(z + Nz*t)))
```

**Note:** This matches plaq's `pack_eo` / `unpack_eo` functions in `src/plaq/layouts.py`.

---

## Gamma Matrix Convention

### plaq Convention (MILC)

plaq uses the **MILC gamma matrix convention** defined in `src/plaq/conventions/gamma_milc.py`:

$$\gamma_0 = \begin{pmatrix} 0 & 0 & 0 & i \\ 0 & 0 & i & 0 \\ 0 & -i & 0 & 0 \\ -i & 0 & 0 & 0 \end{pmatrix}, \quad
\gamma_1 = \begin{pmatrix} 0 & 0 & 0 & -1 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ -1 & 0 & 0 & 0 \end{pmatrix}$$

$$\gamma_2 = \begin{pmatrix} 0 & 0 & i & 0 \\ 0 & 0 & 0 & -i \\ -i & 0 & 0 & 0 \\ 0 & i & 0 & 0 \end{pmatrix}, \quad
\gamma_3 = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix}$$

$$\gamma_5 = \gamma_0 \gamma_1 \gamma_2 \gamma_3 = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}$$

### QUDA Convention (DeGrand-Rossi)

QUDA uses `QUDA_DEGRAND_ROSSI_GAMMA_BASIS` which matches MILC:

From `enum_quda.h`:
```
QUDA_DEGRAND_ROSSI_GAMMA_BASIS:
  gam1 = ((0, i*s1), (-i*s1, 0))
  gam2 = ((0, -i*s2), (i*s2, 0))
  gam3 = ((0, i*s3), (-i*s3, 0))
  gam4 = ((0, 1), (1, 0))
  gam5 = ((1, 0), (0, -1))
```

The MILC convention is equivalent to DeGrand-Rossi with the identification:
- $\gamma_0$ (plaq x-direction) $\leftrightarrow$ `gam1` (QUDA)
- $\gamma_1$ (plaq y-direction) $\leftrightarrow$ `gam2` (QUDA)
- $\gamma_2$ (plaq z-direction) $\leftrightarrow$ `gam3` (QUDA)
- $\gamma_3$ (plaq t-direction) $\leftrightarrow$ `gam4` (QUDA)

---

## Wilson Dirac Operator

### Definition

The Wilson Dirac operator in plaq is defined as:

$$M\psi(x) = (m_0 + 4r)\psi(x) - \frac{1}{2}\sum_{\mu=0}^{3} \left[ (r - \gamma_\mu) U_\mu(x) \psi(x+\hat{\mu}) + (r + \gamma_\mu) U_\mu^\dagger(x-\hat{\mu}) \psi(x-\hat{\mu}) \right]$$

Where:
- $m_0$ is the bare mass parameter
- $r = 1$ is the Wilson parameter
- $U_\mu(x)$ is the gauge link connecting site $x$ to $x + \hat{\mu}$

### Hopping Parameter

The hopping parameter $\kappa$ is related to the mass by:

$$\kappa = \frac{1}{2(m_0 + 4r)}$$

QUDA uses $\kappa$-normalization internally.

### $\gamma_5$-Hermiticity

The Wilson operator satisfies:

$$M^\dagger = \gamma_5 M \gamma_5$$

This is used to compute $M^\dagger$ in plaq and verified by QUDA.

---

## Solver Configuration

### Direct Solve ($Mx = b$)

For solving the direct equation $Mx = b$:

```cpp
inv_param.inv_type = QUDA_BICGSTAB_INVERTER;
inv_param.solution_type = QUDA_MAT_SOLUTION;
inv_param.solve_type = QUDA_DIRECT_SOLVE;
```

### Normal Equation ($M^\dagger M x = M^\dagger b$)

For solving the normal equation:

```cpp
inv_param.inv_type = QUDA_CG_INVERTER;
inv_param.solution_type = QUDA_MATDAG_MAT_SOLUTION;
inv_param.solve_type = QUDA_NORMOP_SOLVE;
```

**Note:** QUDA handles the $M^\dagger b$ multiplication internally when `QUDA_MATDAG_MAT_SOLUTION` is specified.

---

## Boundary Conditions

### Temporal Boundary

plaq uses:
- `bc.fermion_bc_time = -1.0` for **antiperiodic** (default)
- `bc.fermion_bc_time = +1.0` for **periodic**

QUDA uses:
- `QUDA_ANTI_PERIODIC_T` for antiperiodic
- `QUDA_PERIODIC_T` for periodic

The conversion is straightforward:
```cpp
gauge_param.t_boundary = (t_boundary == -1) ? QUDA_ANTI_PERIODIC_T : QUDA_PERIODIC_T;
```

### Spatial Boundaries

Both plaq and QUDA default to **periodic** spatial boundaries. This is handled by the lattice neighbor tables in plaq and QUDA's internal lattice structure.

---

## Memory Layout Summary

| Field Type | plaq Shape | plaq Order | QUDA Shape | QUDA Order |
|------------|------------|------------|------------|------------|
| Gauge      | `[4, V, 3, 3]` | dir, lex-site, row, col | `gauge[4][V, 3, 3]` | dir ptr, eo-site, row, col |
| Spinor     | `[V, 4, 3]` | lex-site, spin, color | `[V, 4, 3]` | eo-site, spin, color |

---

## API Usage Example

```python
import plaq as pq

# Create lattice and fields
lat = pq.Lattice((4, 4, 4, 8))
bc = pq.BoundaryCondition()
lat.build_neighbor_tables(bc)

U = pq.GaugeField.random(lat)
b = pq.SpinorField.random(lat)

# Solve using QUDA backend
params = pq.WilsonParams(mass=0.1)
x, info = pq.solve(U, b, backend="quda", params=params, bc=bc)

print(f"Converged: {info.converged}")
print(f"Iterations: {info.iters}")
print(f"Residual: {info.final_residual}")
```

---

## Data Copy Notes

### Current Implementation

The current implementation requires data copies due to site ordering differences:

| Operation | Copy Required | Reason |
|-----------|---------------|--------|
| Gauge: plaq → QUDA | Yes | Lexicographic → Even-odd site reordering |
| Spinor: plaq → QUDA | Yes | Lexicographic → Even-odd site reordering |
| Spinor: QUDA → plaq | Yes | Even-odd → Lexicographic site reordering |

### Why Copies Are Necessary

1. **Site Ordering Mismatch**: plaq uses lexicographic site ordering for natural array indexing, while QUDA uses even-odd (checkerboard) ordering for optimal GPU cache performance.

2. **GPU Optimization**: QUDA's even-odd ordering enables efficient parallelization by separating even and odd sites, reducing data dependencies in Dslash computations.

### Potential Zero-Copy Path

To achieve zero-copy data transfer, plaq would need to:

1. Store fields in even-odd order natively (see `pack_eo`/`unpack_eo` in `layouts.py`)
2. Pass PyTorch tensor data pointers directly to QUDA

The conversion overhead is typically small compared to the GPU solve time for moderate-to-large lattices, but for repeated solves with the same gauge field, caching the converted gauge field on GPU would eliminate redundant conversions.

### Memory Allocation

```
Gauge field:  4 × V × 9 × 2 × sizeof(T)  bytes
Spinor field: V × 12 × 2 × sizeof(T)     bytes

For 16³×32 lattice with double precision:
  V = 131072 sites
  Gauge:  ~72 MB (temporary copy during solve)
  Spinor: ~24 MB per spinor (source + solution)
```
