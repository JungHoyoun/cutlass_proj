"""
기본 기능 테스트
"""

import pytest
import torch
import numpy as np


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_available():
    """CUDA 사용 가능 여부 확인"""
    assert torch.cuda.is_available()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensor_creation():
    """CUDA 텐서 생성 테스트"""
    device = torch.device('cuda')
    x = torch.randn(10, 10, device=device)
    assert x.is_cuda
    assert x.shape == (10, 10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gemm_basic():
    """기본 GEMM 연산 테스트"""
    device = torch.device('cuda')
    A = torch.randn(1024, 512, device=device)
    B = torch.randn(512, 1024, device=device)
    C = torch.matmul(A, B)
    
    assert C.shape == (1024, 1024)
    assert C.is_cuda


if __name__ == '__main__':
    pytest.main([__file__])

