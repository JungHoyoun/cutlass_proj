/***************************************************************************************************
 * CUTLASS GEMM Python Bindings Header
 * 
 * 주의: CUTLASS 헤더는 CUDA 파일(.cu)에서만 포함해야 합니다.
 * C++ 파일에서는 이 헤더의 함수 선언만 사용합니다.
 **************************************************************************************************/

#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// CUTLASS GEMM 함수 선언 (cutlass_gemm_wrapper.cu에서 구현)
// C++ 파일에서 사용할 수 있도록 선언만 제공
#ifdef __cplusplus
extern "C" {
#endif
cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc);
#ifdef __cplusplus
}
#endif

// PyTorch Tensor를 받는 래퍼 함수
torch::Tensor cutlass_gemm_torch(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    float alpha = 1.0f,
    float beta = 0.0f);

