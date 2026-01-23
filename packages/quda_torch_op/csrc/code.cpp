/**
 * quda_torch_op CPU implementation
 *
 * This file implements the simple_add operator using PyTorch's stable C++ API.
 * The operator performs element-wise addition of two tensors.
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

// Defines the operator
STABLE_TORCH_LIBRARY(quda_torch_op, m) {
  m.def("simple_add(Tensor a, Tensor b) -> Tensor");
}

// Registers CPU implementation
STABLE_TORCH_LIBRARY_IMPL(quda_torch_op, CPU, m) {
  m.impl("simple_add", TORCH_BOX(&simple_add_cpu));
}

}  // namespace quda_torch_op
