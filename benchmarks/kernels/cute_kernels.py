# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
CuTe DSL 커널 구현
"""

import torch


def cute_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    """
    CuTe DSL GEMM 커널
    
    Args:
        A: [m, k] 행렬
        B: [k, n] 행렬
        **kwargs: 추가 인자
    
    Returns:
        [m, n] 결과 행렬
    """
    from src import _cute
    return _cute.gemm(A, B, **kwargs)


def cute_grouped_gemm(
    A_list: list,
    B_list: list,
    **kwargs
) -> list:
    """
    CuTe DSL Grouped GEMM 커널
    
    Args:
        A_list: A 행렬 리스트
        B_list: B 행렬 리스트
        **kwargs: 추가 인자
    
    Returns:
        결과 행렬 리스트
    """
    raise NotImplementedError("Grouped GEMM not yet implemented")
