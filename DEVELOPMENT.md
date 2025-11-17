# 개발 가이드

C++/CUDA와 Python이 함께 있는 프로젝트의 일반적인 개발 워크플로우를 설명합니다.

## 일반적인 개발 방법

### 1. **Editable Install (권장) - 개발 중**

가장 일반적인 방법입니다. 코드를 수정하면 자동으로 반영됩니다.

```bash
# 처음 한 번만 실행
pip install -e .

# 또는
python setup.py develop
```

**장점:**
- 코드 수정 후 재빌드만 하면 됨 (재설치 불필요)
- Python 코드 변경은 즉시 반영
- C++/CUDA 코드 변경은 재빌드 필요

**재빌드 방법:**
```bash
# C++/CUDA 코드 수정 후
python setup.py build_ext --inplace
```

### 2. **In-place Build - 빠른 테스트**

빌드만 하고 설치하지 않는 방법입니다.

```bash
python setup.py build_ext --inplace
```

**장점:**
- 빠른 빌드
- `build/` 디렉토리에 `.so` 파일 생성
- Python에서 직접 import 가능

**사용:**
```python
import sys
sys.path.insert(0, 'build/lib.linux-x86_64-3.8')  # 빌드 경로 추가
from src import _cutlass
```

### 3. **CMake 직접 사용 - 고급 사용자**

복잡한 빌드 설정이 필요할 때 사용합니다.

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## 개발 워크플로우 비교

### 방법 A: Editable Install (가장 일반적)

```bash
# 초기 설정
pip install -e .

# 개발 중
# 1. Python 코드 수정 → 즉시 반영
# 2. C++/CUDA 코드 수정 → 재빌드 필요
python setup.py build_ext --inplace

# 테스트
python example_usage.py
```

**언제 사용:**
- 일반적인 개발
- 프로젝트를 패키지로 관리하고 싶을 때
- 다른 프로젝트에서 import해서 사용할 때

### 방법 B: In-place Build

```bash
# 매번 빌드
python setup.py build_ext --inplace

# 테스트
python example_usage.py
```

**언제 사용:**
- 빠른 프로토타이핑
- 설치 없이 테스트하고 싶을 때
- CI/CD 파이프라인

### 방법 C: CMake 직접 사용

```bash
# CMake 빌드
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Python에서 사용
export PYTHONPATH=$PWD/build:$PYTHONPATH
python example_usage.py
```

**언제 사용:**
- 복잡한 빌드 옵션이 필요할 때
- 디버깅 정보가 많이 필요할 때
- CMake 캐시를 활용하고 싶을 때

## 실제 개발 시나리오

### 시나리오 1: Python 코드만 수정

```python
# src/__init__.py 수정
# → 즉시 반영, 재빌드 불필요
```

### 시나리오 2: C++/CUDA 코드 수정

```bash
# 1. 코드 수정 (예: cutlass_gemm.cpp)
# 2. 재빌드
python setup.py build_ext --inplace

# 3. 테스트
python example_usage.py
```

### 시나리오 3: 새 함수 추가

1. **C++ 래퍼 추가** (`cutlass_gemm.cpp`):
```cpp
torch::Tensor new_function(torch::Tensor input) {
    // 구현
}
```

2. **바인딩 추가** (`bindings.cpp`):
```cpp
m.def("new_function", &new_function, "Description");
```

3. **재빌드**:
```bash
python setup.py build_ext --inplace
```

4. **테스트**:
```python
from src import _cutlass
result = _cutlass.new_function(input)
```

## 프로젝트별 비교

### TransformerEngine 방식
- **개발**: `pip install -e .` (editable install)
- **배포**: wheel 빌드 후 PyPI 배포
- **특징**: 복잡한 CMake 빌드 시스템

### PyTorch Extension 방식
- **개발**: `python setup.py build_ext --inplace`
- **배포**: `python setup.py bdist_wheel`
- **특징**: `torch.utils.cpp_extension` 사용

### 현재 프로젝트 (권장)
- **개발**: `pip install -e .` + `build_ext --inplace` (필요시)
- **배포**: `python setup.py bdist_wheel`
- **특징**: CMake + pybind11 조합

## 빠른 개발을 위한 팁

### 1. 자동 재빌드 스크립트

`scripts/dev_build.sh`:
```bash
#!/bin/bash
python setup.py build_ext --inplace && python example_usage.py
```

사용:
```bash
chmod +x scripts/dev_build.sh
./scripts/dev_build.sh
```

### 2. Watch 모드 (파일 변경 감지)

```bash
# watchdog 설치
pip install watchdog

# 파일 변경 시 자동 빌드
watchmedo shell-command \
    --patterns="*.cpp;*.cu;*.h" \
    --recursive \
    --command='python setup.py build_ext --inplace' \
    src/
```

### 3. 디버그 빌드

```bash
# 디버그 정보 포함
python setup.py build_ext --inplace --debug
```

## 결론

**일반적인 개발 워크플로우:**

1. **초기 설정**: `pip install -e .`
2. **개발 중**:
   - Python 코드: 수정 후 즉시 테스트
   - C++/CUDA 코드: 수정 → `build_ext --inplace` → 테스트
3. **배포**: `python setup.py bdist_wheel`

**가장 실용적인 방법:**
```bash
# 한 번만
pip install -e .

# 개발 중 (C++ 수정 시)
python setup.py build_ext --inplace
```

이 방식이 가장 일반적이고 편리합니다!

