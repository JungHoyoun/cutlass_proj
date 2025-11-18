# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
GEMM 벤치마크

GEMM 문제에 대해 여러 커널 구현(CUTLASS, Triton, CuTe DSL)을 비교합니다.
"""

import torch
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import sys
import os

_benchmarks_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_project_root = os.path.dirname(_benchmarks_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from benchmarks.utils import (
    benchmark_latency_base,
    benchmark_precision_base,
)
from benchmarks.kernels import (
    cutlass_gemm,
    triton_gemm,
    cute_gemm,
)


def pytorch_reference(A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
    """PyTorch 기본 matmul 참조 구현"""
    return torch.matmul(A, B)


def benchmark_latency(
    m: int,
    k: int,
    n: int,
    kernels: List[str] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """
    GEMM latency 벤치마크
    
    Args:
        m, k, n: 행렬 차원
        kernels: 벤치마크할 커널 리스트 (None이면 사용 가능한 모든 커널)
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
        "cutlass": cutlass_gemm,
        "triton": triton_gemm,
        "cute": cute_gemm,
    }
    
    results = []
    
    for kernel_name in kernels:
        if kernel_name not in kernel_functions:
            continue
        
        kernel_fn = kernel_functions[kernel_name]
        
        try:
            result = benchmark_latency_base(
                kernel_fn=kernel_fn,
                reference_fn=pytorch_reference,
                m=m,
                k=k,
                n=n,
                dtype=dtype,
                device=device,
            )
            result["kernel"] = kernel_name
            results.append(result)
        except (RuntimeError, ImportError, ModuleNotFoundError) as e:
            print(f"Skipping {kernel_name}: {e}")
            continue
    
    return results


def benchmark_precision(
    m: int,
    k: int,
    n: int,
    kernels: List[str] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """
    GEMM precision 벤치마크
    
    Args:
        m, k, n: 행렬 차원
        kernels: 벤치마크할 커널 리스트 (None이면 사용 가능한 모든 커널)
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
        "cutlass": cutlass_gemm,
        "triton": triton_gemm,
        "cute": cute_gemm,
    }
    
    results = []
    
    for kernel_name in kernels:
        if kernel_name not in kernel_functions:
            continue
        
        kernel_fn = kernel_functions[kernel_name]
        
        try:
            result = benchmark_precision_base(
                kernel_fn=kernel_fn,
                reference_fn=pytorch_reference,
                m=m,
                k=k,
                n=n,
                dtype=dtype,
                device=device,
            )
            result["kernel"] = kernel_name
            results.append(result)
        except (RuntimeError, ImportError, ModuleNotFoundError) as e:
            print(f"Skipping {kernel_name}: {e}")
            continue
    
    return results


if __name__ == "__main__":
    device = torch.device("cuda")
    
    shapes = [
        (128, 256, 128),
        (512, 1024, 512),
        (1024, 2048, 1024),
        (2048, 4096, 2048),
    ]
    
    latency_results = []
    precision_results = []
    
    print("=" * 60)
    print("GEMM 벤치마크")
    print("=" * 60)
    
    for m, k, n in tqdm(shapes, desc="Shapes"):
        latency_batch = benchmark_latency(m, k, n, dtype=torch.float32, device=device)
        latency_results.extend(latency_batch)
        
        precision_batch = benchmark_precision(m, k, n, dtype=torch.float32, device=device)
        precision_results.extend(precision_batch)
    
    if latency_results:
        df_latency = pd.DataFrame(latency_results)
        df_latency.to_csv("gemm_latency_results.csv", index=False)
        print("\nLatency Results:")
        print(df_latency.to_markdown(index=False))
    
    if precision_results:
        df_precision = pd.DataFrame(precision_results)
        df_precision.to_csv("gemm_precision_results.csv", index=False)
        print("\nPrecision Results:")
        print(df_precision.to_markdown(index=False))

