/**
 * quda_torch_op core implementation
 *
 * This file implements the base operators using PyTorch's stable C++ API.
 * QUDA-specific operators are in quda_interface.cu when QUDA_ENABLED=1.
 */

#include <Python.h>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import will load this .so, triggering the STABLE_TORCH_LIBRARY
     static initializers below to register the operators. */
  PyObject* PyInit__C(void) {
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C",   /* name of module */
        NULL,   /* module documentation */
        -1,     /* per-interpreter state size (-1 = global state) */
        NULL,   /* methods */
    };
    return PyModule_Create(&module_def);
  }
}

namespace quda_torch_op {

/**
 * CPU implementation of simple_add operator.
 *
 * Supports float32, float64, int32, and int64 dtypes.
 */
torch::stable::Tensor simple_add_cpu(
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b) {
  // Validate both tensors are on CPU
  STD_TORCH_CHECK(
      a.device().type() == torch::headeronly::DeviceType::CPU,
      "Tensor 'a' must be on CPU");
  STD_TORCH_CHECK(
      b.device().type() == torch::headeronly::DeviceType::CPU,
      "Tensor 'b' must be on CPU");

  // Validate same dtype
  STD_TORCH_CHECK(
      a.scalar_type() == b.scalar_type(),
      "Tensors must have same dtype");

  // Validate same shape
  STD_TORCH_CHECK(
      a.sizes().equals(b.sizes()),
      "Tensors must have same shape");

  // Make contiguous if needed
  torch::stable::Tensor a_contig = torch::stable::contiguous(a);
  torch::stable::Tensor b_contig = torch::stable::contiguous(b);

  // Create output tensor
  torch::stable::Tensor result = torch::stable::empty_like(a_contig);
  int64_t numel = result.numel();

  // Dispatch based on dtype
  auto dtype = a_contig.scalar_type();

  if (dtype == torch::headeronly::ScalarType::Float) {
    const float* a_ptr = a_contig.const_data_ptr<float>();
    const float* b_ptr = b_contig.const_data_ptr<float>();
    float* out_ptr = result.mutable_data_ptr<float>();
    for (int64_t i = 0; i < numel; i++) {
      out_ptr[i] = a_ptr[i] + b_ptr[i];
    }
  } else if (dtype == torch::headeronly::ScalarType::Double) {
    const double* a_ptr = a_contig.const_data_ptr<double>();
    const double* b_ptr = b_contig.const_data_ptr<double>();
    double* out_ptr = result.mutable_data_ptr<double>();
    for (int64_t i = 0; i < numel; i++) {
      out_ptr[i] = a_ptr[i] + b_ptr[i];
    }
  } else if (dtype == torch::headeronly::ScalarType::Int) {
    const int32_t* a_ptr = a_contig.const_data_ptr<int32_t>();
    const int32_t* b_ptr = b_contig.const_data_ptr<int32_t>();
    int32_t* out_ptr = result.mutable_data_ptr<int32_t>();
    for (int64_t i = 0; i < numel; i++) {
      out_ptr[i] = a_ptr[i] + b_ptr[i];
    }
  } else if (dtype == torch::headeronly::ScalarType::Long) {
    const int64_t* a_ptr = a_contig.const_data_ptr<int64_t>();
    const int64_t* b_ptr = b_contig.const_data_ptr<int64_t>();
    int64_t* out_ptr = result.mutable_data_ptr<int64_t>();
    for (int64_t i = 0; i < numel; i++) {
      out_ptr[i] = a_ptr[i] + b_ptr[i];
    }
  } else {
    STD_TORCH_CHECK(false, "Unsupported dtype for simple_add");
  }

  return result;
}

#if !QUDA_ENABLED
// Stub implementations when QUDA is not available
bool quda_is_available_stub() {
    return false;
}

int64_t quda_get_device_count_stub() {
    return 0;
}

void quda_init_stub(int64_t device) {
    STD_TORCH_CHECK(false, "QUDA is not available. Rebuild with QUDA_HOME set to enable QUDA support.");
}

void quda_finalize_stub() {
    // No-op when QUDA not available
}

bool quda_is_initialized_stub() {
    return false;
}

int64_t quda_get_device_stub() {
    return -1;
}

std::string quda_get_version_stub() {
    return "not available";
}
#endif

// Defines the operator library
STABLE_TORCH_LIBRARY(quda_torch_op, m) {
  // Base operators
  m.def("simple_add(Tensor a, Tensor b) -> Tensor");

  // QUDA interface operators
  m.def("quda_is_available() -> bool");
  m.def("quda_get_device_count() -> int");
  m.def("quda_init(int device) -> ()");
  m.def("quda_finalize() -> ()");
  m.def("quda_is_initialized() -> bool");
  m.def("quda_get_device() -> int");
  m.def("quda_get_version() -> str");
}

// Registers CPU implementation for simple_add
STABLE_TORCH_LIBRARY_IMPL(quda_torch_op, CPU, m) {
  m.impl("simple_add", TORCH_BOX(&simple_add_cpu));
}

#if !QUDA_ENABLED
// Register stub implementations when QUDA is not available
STABLE_TORCH_LIBRARY_IMPL(quda_torch_op, CompositeExplicitAutograd, m) {
  m.impl("quda_is_available", TORCH_BOX(&quda_is_available_stub));
  m.impl("quda_get_device_count", TORCH_BOX(&quda_get_device_count_stub));
  m.impl("quda_init", TORCH_BOX(&quda_init_stub));
  m.impl("quda_finalize", TORCH_BOX(&quda_finalize_stub));
  m.impl("quda_is_initialized", TORCH_BOX(&quda_is_initialized_stub));
  m.impl("quda_get_device", TORCH_BOX(&quda_get_device_stub));
  m.impl("quda_get_version", TORCH_BOX(&quda_get_version_stub));
}
#endif

}  // namespace quda_torch_op
