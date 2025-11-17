"""
유틸리티 함수들
"""

import torch
from typing import Tuple, Optional


def check_cuda_tensor(tensor: torch.Tensor, name: str = "tensor") -> None:
    """CUDA 텐서인지 확인"""
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be on CUDA device")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def get_tensor_info(tensor: torch.Tensor) -> dict:
    """텐서 정보 반환"""
    return {
        'shape': tuple(tensor.shape),
        'dtype': tensor.dtype,
        'device': tensor.device,
        'is_contiguous': tensor.is_contiguous(),
    }


def validate_gemm_inputs(
    A: torch.Tensor,
    B: torch.Tensor,
    C: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """GEMM 입력 검증 및 준비"""
    check_cuda_tensor(A, "A")
    check_cuda_tensor(B, "B")
    
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError("A and B must be 2D tensors")
    
    if A.size(1) != B.size(0):
        raise ValueError(f"Matrix dimensions mismatch: A.shape={A.shape}, B.shape={B.shape}")
    
    if C is not None:
        check_cuda_tensor(C, "C")
        expected_shape = (A.size(0), B.size(1))
        if C.shape != expected_shape:
            raise ValueError(f"C shape mismatch: expected {expected_shape}, got {C.shape}")
    
    # 메모리 연속성 보장
    A = A.contiguous()
    B = B.contiguous()
    if C is not None:
        C = C.contiguous()
    
    return A, B, C

