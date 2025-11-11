#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

namespace py = pybind11;

// 기본 바인딩 모듈
PYBIND11_MODULE(_cutlass, m) {
    m.doc() = "CUTLASS Python bindings for PyTorch";
    
    // 여기에 CUTLASS 커널 바인딩 추가
    // 예: m.def("gemm", &gemm_wrapper, "GEMM operation");
}

