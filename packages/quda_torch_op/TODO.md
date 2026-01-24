# TODO: Future Improvements for QUDA Integration

This document lists potential improvements and optimizations for the QUDA-plaq interface.

## Current Status

The QUDA Wilson solver interface is implemented and ready for use. **However, it requires QUDA to be built with at least one gauge field interface enabled (QDP or MILC recommended).**

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for build requirements.

### Verified Features
- [x] Gauge field format conversion (plaq → QUDA QDP format)
- [x] Spinor field format conversion with even-odd reordering
- [x] Wilson Dslash solver integration
- [x] Support for both M and MdagM equations
- [x] Single and double precision support
- [x] Temporal boundary conditions (periodic/antiperiodic)

### Pending Verification
- [ ] Solution verification against CPU reference (requires QUDA with QDP interface)
- [ ] Performance benchmarking

---

## Performance Improvements

### 1. Zero-Copy Data Transfer
**Priority: High**

Currently, data is copied during format conversion:
```cpp
// Current: Copy with reordering
for (int64_t lex_site = 0; lex_site < V; lex_site++) {
    int64_t eo_site = lex_to_eo[lex_site];
    memcpy(dst + eo_site*24, src + lex_site*24, 24*sizeof(T));
}
```

**Improvement:** Use scatter/gather operations or explore QUDA's ability to accept different orderings:
- Investigate if QUDA can accept lexicographic ordering directly with `QUDA_LEX_DIRAC_ORDER`
- Use pinned memory for faster CPU-GPU transfers
- Consider CUDA unified memory for automatic data movement

### 2. Persistent Gauge Field Storage
**Priority: High**

Currently, gauge field is loaded/freed for each solve:
```cpp
loadGaugeQuda(gauge_quda.data(), &gauge_param);
invertQuda(solution_quda, source_quda, &inv_param);
freeGaugeQuda();
```

**Improvement:** Cache the gauge field on GPU when solving multiple systems with the same gauge:
- Add `load_gauge()` and `free_gauge()` separate from `invert()`
- Use QUDA's `make_resident_gauge` option
- Implement a gauge field cache at the Python level

### 3. GPU-Native Tensor Support
**Priority: Medium**

Currently, input tensors must be on CPU:
```python
if gauge_data.device.type != "cpu":
    gauge_data = gauge_data.cpu()
```

**Improvement:** Accept CUDA tensors directly:
- Use QUDA's `QUDA_CUDA_FIELD_LOCATION` for input/output
- Avoid CPU-GPU roundtrip when data is already on GPU
- Requires careful memory management with PyTorch CUDA tensors

### 4. Asynchronous Execution
**Priority: Medium**

Currently, `wilson_invert()` blocks until completion.

**Improvement:** Add asynchronous solve capability:
- Return a future/promise object
- Allow overlapping computation and data transfer
- Enable pipeline of multiple solves

### 5. Multi-GPU Support
**Priority: Low**

Currently, only single-GPU execution is supported.

**Improvement:** Enable multi-GPU parallelism:
- Expose MPI grid topology configuration
- Support domain decomposition across GPUs
- Integrate with PyTorch's distributed module

---

## Interface Improvements

### 6. Even-Odd Preconditioning
**Priority: High**

Currently, only full-system solve is implemented.

**Improvement:** Add Schur complement preconditioning:
```python
x, info = pq.solve(U, b, backend="quda", precond="eo")
```

Requires:
- Expose `QUDA_MATPC_SOLUTION` solution type
- Handle even/odd parity source construction
- Implement reconstruction of full solution from odd-site solution

### 7. Mixed Precision Solver
**Priority: Medium**

Currently, working precision matches input precision.

**Improvement:** Support mixed-precision acceleration:
```python
x, info = pq.solve(U, b, backend="quda", precision="mixed")
```

QUDA supports different precisions for:
- `cuda_prec` - full precision
- `cuda_prec_sloppy` - inner iterations
- `cuda_prec_precondition` - preconditioner

### 8. Solver Callbacks
**Priority: Low**

Currently, callbacks are not supported for QUDA backend.

**Improvement:** Implement iteration callbacks:
- QUDA supports `reliable_delta` for reliable updates
- Could add custom callback via QUDA's monitoring hooks
- Useful for logging, early stopping, adaptive tolerance

---

## Code Quality Improvements

### 9. Error Handling
**Priority: High**

Currently, QUDA errors may not be properly caught.

**Improvement:** Better error handling:
```cpp
qudaError_t err = qudaGetLastError();
if (err != QUDA_SUCCESS) {
    throw std::runtime_error("QUDA error: " + std::string(qudaGetErrorString(err)));
}
```

### 10. Memory Leak Detection
**Priority: Medium**

Add memory tracking for QUDA allocations:
- Track allocated gauge/spinor memory
- Add cleanup in destructor/finalizer
- Consider RAII wrappers for QUDA resources

### 11. Verbose Mode
**Priority: Low**

Add configurable verbosity:
```python
quda_torch_op.set_verbosity("verbose")  # or "silent", "summarize", "debug"
```

---

## Storage Format Optimizations

### 12. Native plaq EO Format
**Priority: Medium**

Currently, we convert from plaq site layout to QUDA EO layout.

**Improvement:** Accept plaq's EO layout directly:
```python
# If spinor is already in EO layout, skip conversion
spinor_eo = spinor.as_layout("eo")
```

This requires:
- Detecting input layout
- Verifying plaq's EO ordering matches QUDA's
- Zero-copy path for matching formats

### 13. Gauge Field Compression
**Priority: Low**

Currently, full 3x3 matrices are stored.

**Improvement:** Support SU(3) reconstruction:
- `QUDA_RECONSTRUCT_12` - store 12 of 18 real numbers
- `QUDA_RECONSTRUCT_8` - store 8 real numbers
- Reduces memory bandwidth, trades compute for memory

---

## Additional Features

### 14. Clover Wilson Operator
**Priority: Medium**

Add support for clover-improved Wilson fermions:
- Implement `clover_invert()` function
- Handle clover term computation or loading
- Support csw (Sheikholeslami-Wohlert coefficient)

### 15. Eigensolver Interface
**Priority: Low**

Expose QUDA's eigensolver:
```python
eigenvalues, eigenvectors = quda_torch_op.eigsolve(U, n_ev=10)
```

### 16. Multigrid Preconditioner
**Priority: Low**

Add multigrid preconditioning for near-critical masses:
- Expose `newMultigridQuda()` / `destroyMultigridQuda()`
- Configure multigrid levels and parameters
- Significant speedup for light quark masses

---

## Build System Improvements

### 17. CMake Build Support
**Priority: Medium**

Add CMake as alternative to setup.py:
- Better IDE integration
- Easier debugging
- Consistent with QUDA's build system

### 18. Conda Package
**Priority: Low**

Create conda package for easier installation:
- Include QUDA as dependency
- Handle MPI dependency
- Support different CUDA versions

---

## Testing Improvements

### 19. Comprehensive Test Suite
**Priority: High**

Add more tests:
- Different lattice sizes (including non-even dimensions)
- Different mass values (light to heavy)
- Both equation types (M and MdagM)
- Boundary condition variations
- Numerical precision tests

### 20. Benchmarking Suite
**Priority: Medium**

Add performance benchmarks:
- Compare QUDA vs plaq CPU backend
- Memory usage tracking
- Scaling with lattice size
- Multi-precision comparison

---

## Notes

- Items are roughly ordered by priority within each section
- "High" priority items should be addressed before production use
- "Medium" priority items improve usability significantly
- "Low" priority items are nice-to-have features
