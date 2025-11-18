# CUTLASS Python Bindings

# Lazy import to avoid circular import issues
import sys
import importlib

def __getattr__(name):
    if name == '_cutlass':
        # Directly import the module to avoid recursion
        module = importlib.import_module('src._cutlass', package='src')
        return module
    if name == 'gemm':
        module = importlib.import_module('src._cutlass', package='src')
        return module.gemm
    raise AttributeError(f"module 'src' has no attribute '{name}'")

__all__ = ['gemm', '_cutlass']
