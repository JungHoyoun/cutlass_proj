# CUTLASS Python 바인딩

이 프로젝트는 CUTLASS 예제 코드를 Python에서 사용할 수 있도록 Python 바인딩을 제공합니다.

## 구조

TransformerEngine의 방식을 참고하여 다음과 같은 구조로 구현되었습니다:

1. **C++ 래퍼 함수** (`src/bindings/cutlass_gemm.cpp`): PyTorch Tensor를 받아서 CUTLASS 함수를 호출
2. **CUDA 커널 래퍼** (`src/bindings/cutlass_gemm_wrapper.cu`): CUTLASS 예제 코드를 재사용하여 GEMM 연산 구현
3. **pybind11 바인딩** (`src/bindings/bindings.cpp`): Python에서 호출 가능하도록 함수 바인딩
4. **CMake 빌드 시스템**: CUTLASS와 PyTorch를 함께 컴파일

## 빌드 방법

### 1. 의존성 설치

```bash
# PyTorch 설치 (CUDA 지원 버전)
pip install torch torchvision

# pybind11 설치
pip install pybind11

# CMake 설치 (시스템 패키지 매니저 사용)
# Ubuntu/Debian:
sudo apt-get install cmake

# 또는 conda:
conda install cmake
```

### 2. CUTLASS 서브모듈 초기화 (필요한 경우)

```bash
git submodule update --init --recursive
```

### 3. 빌드

```bash
# 개발 모드로 설치 (권장)
pip install -e .

# 또는 직접 빌드
python setup.py build_ext --inplace
```

## 사용 방법

### 기본 사용 예제

```python
import torch
from src import _cutlass

# GPU 텐서 생성
device = torch.device('cuda')
M, K, N = 128, 256, 64

A = torch.randn(M, K, device=device, dtype=torch.float32)
B = torch.randn(K, N, device=device, dtype=torch.float32)
C = torch.zeros(M, N, device=device, dtype=torch.float32)

# CUTLASS GEMM 실행
alpha = 1.0
beta = 0.0
result = _cutlass.gemm(A, B, C, alpha, beta)

# PyTorch 기본 연산과 비교
result_pytorch = torch.matmul(A, B)
print(f"차이: {torch.max(torch.abs(result - result_pytorch))}")
```

### 예제 스크립트 실행

```bash
python example_usage.py
```

## 구현된 기능

현재 구현된 기능:
- ✅ **Basic GEMM**: `3rd_party/cutlass/examples/00_basic_gemm` 기반
  - RowMajor 레이아웃 지원 (PyTorch 호환)
  - float32 데이터 타입
  - alpha, beta 스칼라 지원

## 다른 CUTLASS 예제 추가하기

다른 CUTLASS 예제를 추가하려면:

1. **CUDA 래퍼 파일 생성** (`src/bindings/your_example_wrapper.cu`):
   - CUTLASS 예제 코드의 핵심 함수를 래퍼로 구현

2. **C++ 인터페이스 추가** (`src/bindings/your_example.cpp`):
   - PyTorch Tensor를 받는 래퍼 함수 작성

3. **헤더 파일 추가** (`src/bindings/your_example.h`):
   - 함수 선언

4. **바인딩 추가** (`src/bindings/bindings.cpp`):
   ```cpp
   m.def("your_function", &your_function_torch, "Description");
   ```

5. **CMakeLists.txt 업데이트**:
   ```cmake
   pybind11_add_module(_cutlass
       ...
       src/bindings/your_example.cpp
       src/bindings/your_example_wrapper.cu
   )
   ```

## 주요 파일 설명

- `src/bindings/cutlass_gemm.h`: CUTLASS GEMM 함수 선언
- `src/bindings/cutlass_gemm.cpp`: PyTorch Tensor 래퍼 구현
- `src/bindings/cutlass_gemm_wrapper.cu`: CUTLASS GEMM CUDA 커널 구현
- `src/bindings/bindings.cpp`: pybind11 Python 바인딩
- `CMakeLists.txt`: CMake 빌드 설정
- `setup.py`: Python 패키지 빌드 설정

## 문제 해결

### 빌드 오류

1. **CUTLASS를 찾을 수 없음**:
   ```bash
   git submodule update --init --recursive
   ```

2. **CUDA 아키텍처 오류**:
   `CMakeLists.txt`에서 `CMAKE_CUDA_ARCHITECTURES`를 자신의 GPU에 맞게 수정:
   ```cmake
   set(CMAKE_CUDA_ARCHITECTURES "75")  # 예: RTX 2060
   ```

3. **PyTorch를 찾을 수 없음**:
   PyTorch가 올바르게 설치되어 있는지 확인:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

## 참고 자료

- [CUTLASS 공식 문서](https://github.com/NVIDIA/cutlass)
- [TransformerEngine 구현](https://github.com/NVIDIA/TransformerEngine)
- [pybind11 문서](https://pybind11.readthedocs.io/)

