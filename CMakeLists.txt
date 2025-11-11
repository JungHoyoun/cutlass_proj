cmake_minimum_required(VERSION 3.18)
project(cutlass_python)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)

# PyTorch 찾기
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Cutlass 경로 설정
set(CUTLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/cutlass")
set(CUTLASS_INCLUDE "${CUTLASS_ROOT}/include")
set(CUTLASS_TOOLS "${CUTLASS_ROOT}/tools")

# Cutlass 헤더 파일 확인
if(NOT EXISTS "${CUTLASS_INCLUDE}")
    message(FATAL_ERROR "CUTLASS not found at ${CUTLASS_INCLUDE}. Please initialize git submodule.")
endif()

# Include directories
include_directories(
    ${CUTLASS_INCLUDE}
    ${CUTLASS_TOOLS}/util/include
    ${pybind11_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# CUDA 컴파일러 플래그
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 아키텍처 설정 (사용자 환경에 맞게 조정)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

# Python extension 모듈
pybind11_add_module(_cutlass
    src/cutlass_python/bindings/bindings.cpp
    src/cutlass_python/bindings/kernels.cpp
)

# CUDA 라이브러리 링크
target_link_libraries(_cutlass PRIVATE
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_cudart_LIBRARY}
)

# 컴파일러 플래그
target_compile_options(_cutlass PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler -fPIC
        -gencode arch=compute_70,code=sm_70
        -gencode arch=compute_75,code=sm_75
        -gencode arch=compute_80,code=sm_80
        -gencode arch=compute_86,code=sm_86
        -gencode arch=compute_89,code=sm_89
        -gencode arch=compute_90,code=sm_90
    >
)

# 설치 설정
set_target_properties(_cutlass PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN 1
)

