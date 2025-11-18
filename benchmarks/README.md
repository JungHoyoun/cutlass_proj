# CUTLASS 프로젝트 벤치마크

이 디렉토리는 CUTLASS, Triton, CuTe DSL 등 다양한 커널의 latency와 precision을 벤치마크하는 도구를 제공합니다.

## 구조

```
benchmarks/
├── __init__.py                    # 모듈 초기화
├── utils.py                       # 공통 유틸리티 함수
├── kernels/                       # 커널 구현
│   ├── __init__.py
│   ├── cutlass_kernels.py        # CUTLASS 커널 구현
│   ├── triton_kernels.py         # Triton 커널 구현
│   └── cute_kernels.py           # CuTe DSL 커널 구현
├── problems/                      # 문제별 벤치마크
│   ├── __init__.py
│   ├── benchmark_gemm.py         # GEMM 벤치마크
│   └── benchmark_grouped_gemm.py # Grouped GEMM 벤치마크
└── README.md                     # 이 파일
```

**설계 철학**: 문제(operation)별로 구조화되어 있어, 같은 문제에 대해 여러 커널 구현을 쉽게 비교할 수 있습니다.

## 사용법

### 문제별 벤치마크

각 문제(operation)별로 독립적으로 벤치마크를 실행할 수 있습니다:

```bash
# GEMM 벤치마크 (모든 사용 가능한 커널 비교)
python benchmarks/problems/benchmark_gemm.py

# Grouped GEMM 벤치마크
python benchmarks/problems/benchmark_grouped_gemm.py
```

### Python 코드에서 사용

```python
import torch
from benchmarks.problems.benchmark_gemm import benchmark_latency, benchmark_precision

device = torch.device("cuda")

# Latency 벤치마크 (모든 사용 가능한 커널)
latency_results = benchmark_latency(
    m=1024, k=2048, n=1024, 
    dtype=torch.float32, 
    device=device
)

for result in latency_results:
    print(f"{result['kernel']}: {result['speedup']:.2f}x speedup")

# 특정 커널만 선택
latency_results = benchmark_latency(
    m=1024, k=2048, n=1024,
    kernels=["cutlass", "triton"],  # 특정 커널만 선택
    dtype=torch.float32,
    device=device
)
```

## 새로운 문제 추가하기

새로운 연산(예: batched GEMM, sparse GEMM 등)을 벤치마크하려면:

1. `problems/benchmark_<problem_name>.py` 파일 생성
2. `benchmark_latency()`와 `benchmark_precision()` 함수 구현
3. `kernels/` 디렉토리에 해당 커널 구현이 있다면 사용

예제:

```python
# benchmarks/problems/benchmark_batched_gemm.py
from benchmarks.utils import benchmark_latency_base, benchmark_precision_base
from benchmarks.kernels import cutlass_gemm, triton_gemm, cute_gemm
import torch

def pytorch_reference(A, B, **kwargs):
    return torch.bmm(A, B)  # batched matmul

def benchmark_latency(batch_size, m, k, n, kernels=None, dtype=torch.float32, device=None):
    if device is None:
        device = torch.device("cuda")
    
    kernel_functions = {
        "cutlass": cutlass_gemm,  # batched 버전이 있다면
        "triton": triton_gemm,
        "cute": cute_gemm,
    }
    
    results = []
    for kernel_name in kernels or kernel_functions.keys():
        # 커널별 벤치마크 로직
        ...
    
    return results
```

## 새로운 커널 추가하기

새로운 커널 구현을 추가하려면:

1. `kernels/<kernel_name>_kernels.py` 파일 생성
2. 각 문제에 대한 커널 함수 구현 (예: `gemm()`, `grouped_gemm()`)
3. `kernels/__init__.py`에 export 추가

예제:

```python
# kernels/my_kernel.py
import torch

def my_gemm(A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
    """My custom GEMM kernel"""
    # 커널 구현
    return result

# kernels/__init__.py에 추가
from kernels.my_kernel import my_gemm
```

그러면 모든 문제 벤치마크에서 자동으로 사용 가능한 커널로 인식됩니다.

## 출력 형식

벤치마크 결과는 CSV 파일로 저장되며, 다음 정보를 포함합니다:

### Latency 결과
- `m`, `k`, `n`: 행렬 차원 (또는 문제별 특정 파라미터)
- `dtype`: 데이터 타입
- `kernel`: 커널 이름 (cutlass, triton, cute 등)
- `reference_latency (us)`: 참조 구현 latency (마이크로초)
- `kernel_latency (us)`: 커널 latency (마이크로초)
- `speedup`: 속도 향상 배수

### Precision 결과
- `m`, `k`, `n`: 행렬 차원 (또는 문제별 특정 파라미터)
- `dtype`: 데이터 타입
- `kernel`: 커널 이름
- `error (dB)`: 상대 오차 (dB)

## 요구사항

- PyTorch (CUDA 지원)
- pandas
- tqdm
- triton (Triton 벤치마크용)
- CUTLASS Python 바인딩 (CUTLASS 벤치마크용)

## 참고

- 벤치마크는 중앙값(median)을 사용하여 측정합니다
- 워밍업 25회, 측정 100회를 기본값으로 사용합니다
- Precision 벤치마크는 고정된 시드(42)를 사용하여 재현성을 보장합니다
- 각 문제 벤치마크는 사용 가능한 모든 커널을 자동으로 감지하여 테스트합니다

## 기존 파일 (Deprecated)

다음 파일들은 이전 구조에서 사용되었으나, 새로운 문제별 구조로 대체되었습니다:
- `benchmark_cutlass.py`
- `benchmark_triton.py`
- `benchmark_cute.py`
- `benchmark_all.py`

새로운 구조에서는 `problems/` 디렉토리의 문제별 벤치마크 파일을 사용하세요.
