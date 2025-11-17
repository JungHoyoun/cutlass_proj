/***************************************************************************************************
 * CUTLASS GEMM Wrapper - CUTLASS 예제 코드를 재사용
 * PyTorch는 row-major 레이아웃을 사용하므로 RowMajor 템플릿 사용
 **************************************************************************************************/

#include <cuda_runtime.h>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

// CUTLASS GEMM 함수 선언 (C 링크로 export)
extern "C" {

// CUTLASS basic_gemm.cu의 CutlassSgemmNN 함수 구현
// PyTorch 텐서는 row-major이므로 RowMajor 레이아웃 사용
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
  int ldc) {

  // CUTLASS GEMM 템플릿 정의
  // PyTorch는 row-major이므로 RowMajor 레이아웃 사용
  using RowMajor = cutlass::layout::RowMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                  RowMajor,     // Layout of A matrix (row-major)
                                                  float,        // Data-type of B matrix
                                                  RowMajor,     // Layout of B matrix (row-major)
                                                  float,        // Data-type of C matrix
                                                  RowMajor>;    // Layout of C matrix (row-major)

  // CUTLASS GEMM 연산자 생성
  CutlassGemm gemm_operator;

  // CUTLASS GEMM 인자 구성
  // RowMajor의 경우 leading dimension은 열의 개수입니다
  CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D
                              {alpha, beta}); // Scalars used in the Epilogue

  // CUTLASS GEMM 커널 실행
  cutlass::Status status = gemm_operator(args);

  // 에러 체크
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}
}  // extern "C"

