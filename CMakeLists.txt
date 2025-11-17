cmake_minimum_required(VERSION 3.18)

project(cutlass_proj)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Cutlass 경로 설정
# CMAKE_CURRENT_SOURCE_DIR: 현재 CMakeLists.txt가 있는 디렉토리 경로
set(CUTLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/cutlass")  # Cutlass 루트 디렉토리
set(CUTLASS_INCLUDE "${CUTLASS_ROOT}/include")                      # Cutlass 헤더 파일 디렉토리
set(CUTLASS_TOOLS "${CUTLASS_ROOT}/tools")                          # Cutlass 도구 디렉토리

# include_directories: 컴파일러가 헤더 파일을 찾을 수 있는 디렉토리 경로 추가
# 이 디렉토리들은 #include <...> 또는 #include "..." 구문에서 검색됩니다
include_directories(
    ${CUTLASS_INCLUDE}                                   # Cutlass 메인 헤더
    ${CUTLASS_TOOLS}/util/include                        # Cutlass 유틸리티 헤더
    ${CUTLASS_ROOT}/examples/common                      # CUTLASS 예제 공통 헤더
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings             # 바인딩 헤더
    ${pybind11_INCLUDE_DIRS}                             # pybind11 헤더 디렉토리 (find_package로 자동 설정됨)
    ${CUDA_INCLUDE_DIRS}                                 # CUDA 헤더 디렉토리 (find_package로 자동 설정됨)
    ${TORCH_INCLUDE_DIRS}                                # PyTorch 헤더 디렉토리 (find_package로 자동 설정됨)
)

# CUDA 언어 활성화
# enable_language: 프로젝트에서 사용할 프로그래밍 언어를 활성화
# CUDA를 활성화하면 .cu 파일을 CUDA 컴파일러(nvcc)로 컴파일할 수 있습니다
enable_language(CUDA)

# CUDA 컴파일러 플래그
# CUDA 코드에 사용할 C++ 표준 버전 설정 (C++17)
set(CMAKE_CUDA_STANDARD 17)
# CUDA 표준이 필수임을 명시
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 아키텍처 설정 (사용자 환경에 맞게 조정)
# CUDA 아키텍처: GPU의 compute capability를 의미합니다
# 숫자 의미: 70=Volta, 75=Turing, 80=Ampere, 86=Ampere(모바일), 89=Ada, 90=Hopper
# DEFINED: 변수가 정의되어 있는지 확인
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # 세미콜론(;)으로 구분된 여러 아키텍처를 지정
    # 이렇게 하면 여러 GPU 아키텍처에 대해 코드를 생성합니다 (fat binary)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

# Python extension 모듈
# pybind11_add_module: Python 확장 모듈을 생성하는 pybind11 전용 함수
# 첫 번째 인자(_cutlass): 생성될 Python 모듈 이름 (import _cutlass로 사용)
# 나머지 인자들: 모듈을 구성하는 소스 파일들
# pybind11은 CUDA 파일(.cu)을 자동으로 감지하고 적절히 처리합니다
pybind11_add_module(_cutlass
    src/bindings/bindings.cpp              # Python 바인딩 코드
    src/bindings/cutlass_gemm.cpp          # C++ 구현 파일
    src/bindings/cutlass_gemm_wrapper.cu   # CUDA 구현 파일
)

# CUDA 라이브러리 링크
# target_link_libraries: 특정 타겟(여기서는 _cutlass 모듈)에 라이브러리를 링크
# PRIVATE: 링크된 라이브러리는 이 타겟에만 적용되고, 이 타겟을 사용하는 다른 타겟에는 전파되지 않음
# PUBLIC: 다른 타겟에도 전파됨, INTERFACE: 다른 타겟에만 전파됨
target_link_libraries(_cutlass PRIVATE
    ${TORCH_LIBRARIES}           # PyTorch 라이브러리들 (find_package로 자동 설정됨)
    ${CUDA_LIBRARIES}            # CUDA 기본 라이브러리들
    ${CUDA_cublas_LIBRARY}       # cuBLAS 라이브러리 (선형대수 연산용)
    ${CUDA_cudart_LIBRARY}       # CUDA 런타임 라이브러리
)

# PyTorch Python 바인딩 라이브러리 추가 (pybind11 Tensor 지원)
# Python을 통해 torch_python 라이브러리 경로 찾기
# torch_python.so는 PyTorch의 Python 바인딩을 위한 라이브러리로,
# pybind11에서 PyTorch Tensor를 사용할 때 필요합니다
execute_process(
    COMMAND python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libtorch_python.so'))"
    OUTPUT_VARIABLE TORCH_PYTHON_LIB      # 찾은 라이브러리 경로를 저장
    OUTPUT_STRIP_TRAILING_WHITESPACE      # 출력 끝의 공백 제거
    ERROR_QUIET                            # 에러를 조용히 처리
)
# 라이브러리 파일이 존재하는지 확인
if(TORCH_PYTHON_LIB AND EXISTS "${TORCH_PYTHON_LIB}")
    target_link_libraries(_cutlass PRIVATE "${TORCH_PYTHON_LIB}")  # 라이브러리 링크
    message(STATUS "Added torch_python library: ${TORCH_PYTHON_LIB}")  # 성공 메시지
else()
    # WARNING: 경고 메시지 출력 (오류는 아니지만 기능이 제한될 수 있음)
    message(WARNING "torch_python library not found. Some PyTorch Tensor operations may not work.")
endif()

# 컴파일러 플래그
# target_compile_options: 특정 타겟에 대한 컴파일 옵션 설정
# $<$<COMPILE_LANGUAGE:CUDA>:...>: CUDA 파일을 컴파일할 때만 적용되는 옵션
# 이는 generator expression으로, 조건부로 옵션을 적용합니다
target_compile_options(_cutlass PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:                    # CUDA 파일에만 적용
        -Xcompiler -fPIC                           # -Xcompiler: nvcc에게 다음 옵션을 호스트 컴파일러에 전달
                                                   # -fPIC: Position Independent Code 생성 (공유 라이브러리용)
        -gencode arch=compute_89,code=sm_89        # compute capability 8.9 (Ada 아키텍처)용 코드 생성
    >
)

# 설치 설정
# set_target_properties: 타겟의 속성을 설정
# CXX_VISIBILITY_PRESET "hidden": 기본적으로 심볼을 숨김 (공유 라이브러리에서 불필요한 심볼 노출 방지)
# VISIBILITY_INLINES_HIDDEN 1: 인라인 함수도 숨김 처리
# 이 설정들은 공유 라이브러리의 크기를 줄이고, 심볼 충돌을 방지하는 데 도움이 됩니다
set_target_properties(_cutlass PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN 1
)

