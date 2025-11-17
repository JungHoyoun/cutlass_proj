#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include "cutlass_gemm.h"

namespace py = pybind11;

// 기본 바인딩 모듈
PYBIND11_MODULE(_cutlass, m) {
    m.doc() = "CUTLASS Python bindings for PyTorch";
    
    // CUTLASS GEMM 바인딩
    m.def("gemm", &cutlass_gemm_torch, 
          "CUTLASS GEMM operation",
          py::arg("A"), 
          py::arg("B"), 
          py::arg("C"),
          py::arg("alpha") = 1.0f,
          py::arg("beta") = 0.0f);
}

