# CUTLASS & CUTE_dsl

A project for developing and learning custom GPU kernels by integrating PyTorch with CUTLASS.

## Project Structure

```
cutlass_python/
├── 3rd_party/
│   └── cutlass/              # CUTLASS submodule
├── src/
│   └── cutlass_python/       # Main package
│       ├── __init__.py
│       ├── kernels/          # CUTLASS kernel wrappers
│       ├── examples/         # Example implementations
│       └── bindings/         # C++ bindings
├── tests/                    # Test code
├── scripts/                  # Build scripts
├── Dockerfile
├── CMakeLists.txt
├── setup.py
├── pyproject.toml
└── README.md
```

## Installation

### Using Docker (Recommended)

```bash
bash scripts/build.sh
docker exec -it cutlass_python_dev bash
```

## Usage Example

```python
import torch
import cutlass_python

# Matrix multiplication using CUTLASS kernel
a = torch.randn(1024, 1024, device='cuda')
b = torch.randn(1024, 1024, device='cuda')
c = cutlass_python.gemm(a, b)
```

## Requirements

- Python >= 3.7
- PyTorch >= 1.9.0
- CUDA >= 11.0
- CMake >= 3.18
- CUDA-capable GPU

## Build Options

Specify CUDA architectures:

```bash
export CMAKE_CUDA_ARCHITECTURES="89;90a"
pip install --no-build-isolatio -e .
```

## Benchmarking

### 기본 벤치마크 실행

```bash
# GEMM 벤치마크 실행
python benchmarks/problems/benchmark_gemm.py

# Grouped GEMM 벤치마크 실행
python benchmarks/problems/benchmark_grouped_gemm.py
```

### Nsight Compute로 프로파일링

Nsight Compute를 사용하여 GPU 커널의 상세한 성능 분석을 수행할 수 있습니다:

```bash
# 기본 프로파일링 (모든 메트릭)
nsys profile --trace=cuda,nvtx --output=gemm_profile python benchmarks/problems/benchmark_gemm.py

# Nsight Compute로 특정 커널 프로파일링
ncu --set full \
    --target-processes all \
    --kernel-regex ".*gemm.*" \
    python benchmarks/problems/benchmark_gemm.py

# 특정 메트릭만 수집
ncu --set default \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__sass_thread_inst_executed_op_fp32_pred_on.sum \
    python benchmarks/problems/benchmark_gemm.py

# 결과를 CSV로 저장
ncu --set default --csv --log-file gemm_metrics.csv \
    python benchmarks/problems/benchmark_gemm.py
```

#### 주요 Nsight Compute 옵션

- `--set full`: 모든 메트릭 수집
- `--set default`: 기본 메트릭만 수집 (더 빠름)
- `--kernel-regex`: 특정 커널만 프로파일링
- `--target-processes all`: 모든 프로세스 타겟팅
- `--csv`: CSV 형식으로 출력
- `--log-file`: 결과를 파일로 저장

#### 유용한 메트릭 세트

```bash
# 메모리 대역폭 분석
ncu --set memory \
    python benchmarks/problems/benchmark_gemm.py

# 연산 처리량 분석
ncu --set compute \
    python benchmarks/problems/benchmark_gemm.py

# 워프 효율성 분석
ncu --set warp \
    python benchmarks/problems/benchmark_gemm.py
```

자세한 벤치마크 사용법은 `benchmarks/README.md`를 참고하세요.