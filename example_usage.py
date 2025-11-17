#!/usr/bin/env python3
"""
CUTLASS GEMM Python 사용 예제
"""

import torch
import sys
import os

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src import _cutlass
    print("CUTLASS 모듈 로드 성공!")
except ImportError as e:
    print(f"CUTLASS 모듈을 찾을 수 없습니다: {e}")
    print("먼저 'python setup.py build_ext --inplace' 또는 'pip install -e .'를 실행하세요.")
    sys.exit(1)

def test_cutlass_gemm():
    """CUTLASS GEMM 테스트"""
    
    # GPU 확인
    if not torch.cuda.is_available():
        print("CUDA를 사용할 수 없습니다.")
        return
    
    device = torch.device('cuda')
    
    # 테스트 행렬 생성
    M, K, N = 128, 256, 64
    alpha = 1.0
    beta = 0.0
    
    print(f"GEMM 테스트: A[{M}x{K}] @ B[{K}x{N}] = C[{M}x{N}]")
    
    # 행렬 초기화
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)
    C = torch.zeros(M, N, device=device, dtype=torch.float32)
    
    print("PyTorch 기본 matmul 실행...")
    result_pytorch = torch.matmul(A, B)
    
    print("CUTLASS GEMM 실행...")
    result_cutlass = _cutlass.gemm(A, B, C, alpha, beta)
    
    # 결과 비교
    diff = torch.abs(result_pytorch - result_cutlass)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"\n결과 비교:")
    print(f"  최대 차이: {max_diff:.6f}")
    print(f"  평균 차이: {mean_diff:.6f}")
    
    if max_diff < 1e-5:
        print("✓ CUTLASS GEMM 결과가 PyTorch와 일치합니다!")
    else:
        print("✗ CUTLASS GEMM 결과가 PyTorch와 다릅니다.")
        print(f"  PyTorch 결과 샘플: {result_pytorch[0, :5]}")
        print(f"  CUTLASS 결과 샘플: {result_cutlass[0, :5]}")

if __name__ == "__main__":
    test_cutlass_gemm()

