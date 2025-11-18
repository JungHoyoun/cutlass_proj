# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Grouped GEMM 벤치마크

Grouped GEMM 문제에 대해 여러 커널 구현을 비교합니다.
"""

import torch
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
import sys
import os

# 프로젝트 루트를 path에 추가
_benchmarks_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_project_root = os.path.dirname(_benchmarks_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from benchmarks.utils import (
    benchmark_microseconds,
    compute_error,
    get_problem,
)
from benchmarks.kernels import (
    cutlass_grouped_gemm,
    triton_grouped_gemm,
    cute_grouped_gemm,
)

if not torch.cuda.is_available():
    raise RuntimeError("This benchmark requires CUDA hardware")


def pytorch_grouped_gemm_reference(
    A_list: List[torch.Tensor],
    B_list: List[torch.Tensor],
    **kwargs
) -> List[torch.Tensor]:
    """PyTorch 기본 grouped GEMM 참조 구현"""
    return [torch.matmul(A, B) for A, B in zip(A_list, B_list)]


def get_grouped_gemm_problem(
    num_groups: int,
    shapes: List[Tuple[int, int, int]],
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Grouped GEMM 문제 생성
    
    Args:
        num_groups: 그룹 수
        shapes: 각 그룹의 (m, k, n) shape 리스트
        dtype: 데이터 타입
        device: 디바이스
        seed: 랜덤 시드
    
    Returns:
        (A_list, B_list) 튜플
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if seed is not None:
        torch.manual_seed(seed)
    
    A_list = []
    B_list = []
    
    for m, k, n in shapes:
        A = torch.randn(m, k, dtype=dtype, device=device)
        B = torch.randn(k, n, dtype=dtype, device=device)
        A_list.append(A)
        B_list.append(B)
    
    return A_list, B_list


def benchmark_latency(
    shapes: List[Tuple[int, int, int]],
    kernels: List[str] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """
    Grouped GEMM latency 벤치마크
    
    Args:
        shapes: 각 그룹의 (m, k, n) shape 리스트
        kernels: 벤치마크할 커널 리스트
        dtype: 데이터 타입
        device: 디바이스
    
    Returns:
        벤치마크 결과 리스트
    """
    if device is None:
        device = torch.device("cuda")
    
    if kernels is None:
        kernels = ["cutlass", "triton", "cute"]
    
    kernel_functions = {
        "cutlass": cutlass_grouped_gemm,
        "triton": triton_grouped_gemm,
        "cute": cute_grouped_gemm,
    }
    
    results = []
    num_groups = len(shapes)
    
    # 참조 구현 벤치마크
    A_ref_list, B_ref_list = get_grouped_gemm_problem(
        num_groups, shapes, dtype, device
    )
    reference_time = benchmark_microseconds(
        pytorch_grouped_gemm_reference, A_ref_list, B_ref_list
    )
    
    for kernel_name in kernels:
        if kernel_name not in kernel_functions:
            continue
        
        kernel_fn = kernel_functions[kernel_name]
        
        A_list, B_list = get_grouped_gemm_problem(
            num_groups, shapes, dtype, device
        )
        kernel_time = benchmark_microseconds(kernel_fn, A_list, B_list)
        
        result = {
            "num_groups": num_groups,
            "shapes": str(shapes),
            "dtype": str(dtype),
            "kernel": kernel_name,
            "reference_latency (us)": reference_time,
            "kernel_latency (us)": kernel_time,
            "speedup": reference_time / kernel_time if kernel_time > 0 else float("inf"),
        }
        results.append(result)
    
    return results


def benchmark_precision(
    shapes: List[Tuple[int, int, int]],
    kernels: List[str] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """
    Grouped GEMM precision 벤치마크
    
    Args:
        shapes: 각 그룹의 (m, k, n) shape 리스트
        kernels: 벤치마크할 커널 리스트
        dtype: 데이터 타입
        device: 디바이스
    
    Returns:
        벤치마크 결과 리스트
    """
    if device is None:
        device = torch.device("cuda")
    
    if kernels is None:
        kernels = ["cutlass", "triton", "cute"]
    
    kernel_functions = {
        "cutlass": cutlass_grouped_gemm,
        "triton": triton_grouped_gemm,
        "cute": cute_grouped_gemm,
    }
    
    results = []
    num_groups = len(shapes)
    
    # 참조 결과 계산
    A_ref_list, B_ref_list = get_grouped_gemm_problem(
        num_groups, shapes, dtype, device, seed=42
    )
    reference_output = pytorch_grouped_gemm_reference(A_ref_list, B_ref_list)
    
    for kernel_name in kernels:
        if kernel_name not in kernel_functions:
            continue
        
        kernel_fn = kernel_functions[kernel_name]
        
        A_list, B_list = get_grouped_gemm_problem(
            num_groups, shapes, dtype, device, seed=42
        )
        kernel_output = kernel_fn(A_list, B_list)
        
        # 각 그룹별 오차 계산 후 평균
        errors = []
        for ref_out, kernel_out in zip(reference_output, kernel_output):
            error_db = compute_error(ref_out, kernel_out, relative=True)
            errors.append(error_db)
        
        avg_error = sum(errors) / len(errors) if errors else float("inf")
        
        result = {
            "num_groups": num_groups,
            "shapes": str(shapes),
            "dtype": str(dtype),
            "kernel": kernel_name,
            "error (dB)": avg_error,
        }
        results.append(result)
    
    return results


if __name__ == "__main__":
    device = torch.device("cuda")
    
    # 벤치마크 설정
    # 각 그룹의 shape를 정의
    test_cases = [
        # (num_groups, shapes)
        (4, [(128, 256, 128), (256, 512, 256), (512, 1024, 512), (1024, 2048, 1024)]),
        (8, [(128, 256, 128)] * 8),
    ]
    
    latency_results = []
    precision_results = []
    
    print("=" * 60)
    print("Grouped GEMM 벤치마크")
    print("=" * 60)
    
    for num_groups, shapes in tqdm(test_cases, desc="Test cases"):
        # Latency 벤치마크
        latency_batch = benchmark_latency(shapes, dtype=torch.float32, device=device)
        latency_results.extend(latency_batch)
        
        # Precision 벤치마크
        precision_batch = benchmark_precision(shapes, dtype=torch.float32, device=device)
        precision_results.extend(precision_batch)
    
    # 결과 저장
    if latency_results:
        df_latency = pd.DataFrame(latency_results)
        df_latency.to_csv("grouped_gemm_latency_results.csv", index=False)
        print("\nLatency Results:")
        print(df_latency.to_markdown(index=False))
    
    if precision_results:
        df_precision = pd.DataFrame(precision_results)
        df_precision.to_csv("grouped_gemm_precision_results.csv", index=False)
        print("\nPrecision Results:")
        print(df_precision.to_markdown(index=False))

