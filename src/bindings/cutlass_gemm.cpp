/***************************************************************************************************
 * CUTLASS GEMM Python Bindings Implementation
 **************************************************************************************************/

#include "cutlass_gemm.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// CUTLASS GEMM 구현을 basic_gemm.cu에서 가져옴
// 여기서는 선언만 하고, 실제 구현은 basic_gemm_wrapper.cu에 있음

torch::Tensor cutlass_gemm_torch(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    float alpha,
    float beta) {
    
    // 입력 검증
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    
    // 행렬 차원 확인
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(C.dim() == 2, "C must be 2D");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    TORCH_CHECK(B.size(0) == K, "B must have shape [K, N]");
    TORCH_CHECK(C.size(0) == M, "C must have shape [M, N]");
    TORCH_CHECK(C.size(1) == N, "C must have shape [M, N]");
    
    // CUDA 스트림 가져오기
    c10::cuda::CUDAGuard guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // 결과 텐서 생성 (C와 같은 크기)
    torch::Tensor result = C.clone();
    
    // CUTLASS는 RowMajor 템플릿을 사용하므로
    // PyTorch row-major 텐서의 데이터 포인터를 그대로 사용할 수 있습니다.
    // RowMajor의 경우 leading dimension은 열의 개수입니다.
    int lda = K;  // A는 [M, K] 행렬, leading dimension = K
    int ldb = N;  // B는 [K, N] 행렬, leading dimension = N
    int ldc = N;  // C는 [M, N] 행렬, leading dimension = N
    
    cudaError_t status = CutlassSgemmNN(
        M, N, K,
        alpha,
        A.data_ptr<float>(),
        lda,
        B.data_ptr<float>(),
        ldb,
        beta,
        result.data_ptr<float>(),
        ldc
    );
    
    TORCH_CHECK(status == cudaSuccess, 
                "CUTLASS GEMM failed: ", cudaGetErrorString(status));
    
    return result;
}

