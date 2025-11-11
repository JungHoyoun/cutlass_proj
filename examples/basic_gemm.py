"""
기본 GEMM 예제
"""

import torch
import cutlass_python


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return
    
    device = torch.device('cuda')
    
    # 입력 행렬 생성
    M, N, K = 1024, 1024, 1024
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)
    
    # CUTLASS GEMM 실행
    try:
        C = cutlass_python.kernels.gemm(A, B)
        print(f"GEMM 결과 shape: {C.shape}")
        print(f"GEMM 결과 dtype: {C.dtype}")
        
        # PyTorch 기본 구현과 비교
        C_ref = torch.matmul(A, B)
        max_diff = (C - C_ref).abs().max().item()
        print(f"PyTorch와의 최대 차이: {max_diff}")
        
    except AttributeError:
        print("CUTLASS 커널이 아직 구현되지 않았습니다.")
        print("PyTorch 기본 구현 사용:")
        C = torch.matmul(A, B)
        print(f"결과 shape: {C.shape}")


if __name__ == '__main__':
    main()

