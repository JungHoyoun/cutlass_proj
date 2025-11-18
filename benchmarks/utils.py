# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
벤치마크 공통 유틸리티 함수들
"""

import torch
from typing import Tuple, Optional, Callable, Any
import math
from triton.testing import do_bench


def benchmark_microseconds(
    f: Callable, *args, warmup: int = 25, rep: int = 100, **kwargs
) -> float:
    """
    함수 실행 시간을 마이크로초 단위로 측정
    
    Args:
        f: 벤치마크할 함수
        *args: 함수에 전달할 위치 인자
        warmup: 워밍업 반복 횟수
        rep: 측정 반복 횟수
        **kwargs: 함수에 전달할 키워드 인자
    
    Returns:
        중앙값 실행 시간 (마이크로초)
    """
    if do_bench is not None:
        return do_bench(
            lambda: f(*args, **kwargs), warmup=warmup, rep=rep, return_mode="median"
        ) * 1e3
    else:
        # triton이 없는 경우 torch.cuda.synchronize() 사용
        torch.cuda.synchronize()
        for _ in range(warmup):
            f(*args, **kwargs)
        torch.cuda.synchronize()
        
        times = []
        for _ in range(rep):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            f(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000)  # ms to microseconds
        
        times.sort()
        return times[len(times) // 2]  # median


def compute_error(
    reference: torch.Tensor, 
    computed: torch.Tensor, 
    relative: bool = True
) -> float:
    """
    참조 텐서와 계산된 텐서 간의 오차를 계산
    
    Args:
        reference: 참조 텐서
        computed: 계산된 텐서
        relative: True면 상대 오차(dB), False면 절대 오차
    
    Returns:
        오차 값 (dB 또는 절대값)
        
    Note:
        dB 값이 음수이고 절댓값이 클수록 오차가 작습니다.
        예: -60 dB ≈ 0.1% 오차, -70 dB ≈ 0.03% 오차, -80 dB ≈ 0.01% 오차
    """
    if reference.shape != computed.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference.shape} vs computed {computed.shape}"
        )
    
    diff = torch.abs(reference - computed)
    abs_error = torch.mean(diff).item()
    
    if relative:
        ref_norm = torch.abs(reference).mean().item()
        if ref_norm < 1e-8:
            return float("inf")
        relative_error = abs_error / ref_norm
        # dB 변환: 20 * log10(relative_error)
        if relative_error < 1e-10:
            return -200.0  # 매우 작은 오차
        return 20.0 * math.log10(relative_error)
    else:
        return abs_error


def get_problem(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GEMM 문제 생성 (행렬 A, B)
    
    Args:
        m: A 행렬의 행 수
        n: B 행렬의 열 수
        k: A 행렬의 열 수 (B 행렬의 행 수)
        dtype: 텐서 데이터 타입
        device: 텐서가 위치할 디바이스
        seed: 랜덤 시드 (재현성을 위해)
    
    Returns:
        (A, B) 튜플: [m, k] 행렬 A와 [k, n] 행렬 B
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if seed is not None:
        torch.manual_seed(seed)
    
    A = torch.randn(m, k, dtype=dtype, device=device)
    B = torch.randn(k, n, dtype=dtype, device=device)
    
    return A, B


def benchmark_latency_base(
    kernel_fn: Callable,
    reference_fn: Callable,
    m: int,
    k: int,
    n: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    kernel_kwargs: Optional[dict] = None,
    reference_kwargs: Optional[dict] = None,
) -> dict:
    """
    커널의 latency를 벤치마크하는 기본 함수
    
    Args:
        kernel_fn: 벤치마크할 커널 함수
        reference_fn: 참조 함수 (PyTorch 기본 구현 등)
        m, k, n: 행렬 차원
        dtype: 데이터 타입
        device: 디바이스
        kernel_kwargs: 커널 함수에 전달할 추가 키워드 인자
        reference_kwargs: 참조 함수에 전달할 추가 키워드 인자
    
    Returns:
        벤치마크 결과 딕셔너리
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    kernel_kwargs = kernel_kwargs or {}
    reference_kwargs = reference_kwargs or {}
    
    # 참조 구현 벤치마크
    A_ref, B_ref = get_problem(m, n, k, dtype, device)
    reference_time = benchmark_microseconds(reference_fn, A_ref, B_ref, **reference_kwargs)
    
    # 커널 벤치마크
    A, B = get_problem(m, n, k, dtype, device)
    kernel_time = benchmark_microseconds(kernel_fn, A, B, **kernel_kwargs)
    
    return {
        "m": m,
        "k": k,
        "n": n,
        "dtype": str(dtype),
        "reference_latency (us)": reference_time,
        "kernel_latency (us)": kernel_time,
        "speedup": reference_time / kernel_time if kernel_time > 0 else float("inf"),
    }


def benchmark_precision_base(
    kernel_fn: Callable,
    reference_fn: Callable,
    m: int,
    k: int,
    n: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    kernel_kwargs: Optional[dict] = None,
    reference_kwargs: Optional[dict] = None,
) -> dict:
    """
    커널의 precision을 벤치마크하는 기본 함수
    
    Args:
        kernel_fn: 벤치마크할 커널 함수
        reference_fn: 참조 함수
        m, k, n: 행렬 차원
        dtype: 데이터 타입
        device: 디바이스
        kernel_kwargs: 커널 함수에 전달할 추가 키워드 인자
        reference_kwargs: 참조 함수에 전달할 추가 키워드 인자
    
    Returns:
        벤치마크 결과 딕셔너리
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    kernel_kwargs = kernel_kwargs or {}
    reference_kwargs = reference_kwargs or {}
    
    # 참조 결과 계산
    A, B = get_problem(m, n, k, dtype, device, seed=42)
    reference_output = reference_fn(A, B, **reference_kwargs)
    
    # 커널 결과 계산
    A_kernel, B_kernel = get_problem(m, n, k, dtype, device, seed=42)
    kernel_output = kernel_fn(A_kernel, B_kernel, **kernel_kwargs)
    
    error_db = compute_error(reference_output, kernel_output, relative=True)
    
    # 상대 오차를 퍼센트로도 계산 (더 직관적)
    relative_error = 10 ** (error_db / 20.0) if error_db > -200 else 0.0
    error_percent = relative_error * 100
    
    return {
        "m": m,
        "k": k,
        "n": n,
        "dtype": str(dtype),
        "error (dB)": error_db,
        "error (%)": error_percent,
    }

