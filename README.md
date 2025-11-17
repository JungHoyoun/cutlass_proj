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