# CUTLASS Python

PyTorch와 CUTLASS 예제들을 연동하는 프로젝트입니다. CUTLASS의 다양한 고성능 커널 예제들을 PyTorch에서 쉽게 사용할 수 있도록 래퍼를 제공합니다.

## 프로젝트 구조

```
cutlass_python/
├── 3rd_party/
│   └── cutlass/              # CUTLASS 서브모듈
├── src/
│   └── cutlass_python/
│       ├── __init__.py
│       ├── kernels/          # CUTLASS 커널 래퍼들
│       ├── examples/         # CUTLASS 예제 구현들
│       └── utils.py
├── tests/                    # 테스트 코드
├── examples/                 # 사용 예제
├── scripts/                  # 빌드 및 설치 스크립트
├── CMakeLists.txt
├── setup.py
├── pyproject.toml
└── README.md
```

## 설치 방법

### 1. 서브모듈 초기화

```bash
git submodule update --init --recursive
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 빌드 및 설치

```bash
pip install -e .
```

또는 개발 모드로 설치:

```bash
python setup.py develop
```

## 사용 예제

```python
import torch
import cutlass_python

# CUTLASS 커널을 사용한 행렬 곱셈
a = torch.randn(1024, 1024, device='cuda')
b = torch.randn(1024, 1024, device='cuda')
c = cutlass_python.gemm(a, b)
```

## 지원하는 CUTLASS 예제

- [ ] GEMM (General Matrix Multiply)
- [ ] Conv2D
- [ ] Batch GEMM
- [ ] Grouped GEMM
- [ ] Fused operations
- [ ] 기타 CUTLASS 예제들

## 요구사항

- Python >= 3.7
- PyTorch >= 1.9.0
- CUDA >= 11.0
- CMake >= 3.18
- CUDA-capable GPU

## 빌드 옵션

CUDA 아키텍처를 지정하려면:

```bash
export CMAKE_CUDA_ARCHITECTURES="70;75;80;86"
pip install -e .
```

## 개발

테스트 실행:

```bash
pytest tests/
```

## 라이선스

이 프로젝트는 CUTLASS의 라이선스를 따릅니다.

