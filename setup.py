from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import os
import subprocess
import sys

# 프로젝트 루트 경로
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Cutlass 경로
CUTLASS_ROOT = os.path.join(PROJECT_ROOT, '3rd_party', 'cutlass')
CUTLASS_INCLUDE = os.path.join(CUTLASS_ROOT, 'include')
CUTLASS_TOOLS = os.path.join(CUTLASS_ROOT, 'tools')

# PyTorch 경로 찾기
try:
    import torch
    TORCH_PATH = os.path.dirname(torch.__file__)
    TORCH_LIB = os.path.join(TORCH_PATH, 'lib')
    TORCH_INCLUDE = os.path.join(TORCH_PATH, 'include')
except ImportError:
    TORCH_PATH = None
    TORCH_LIB = None
    TORCH_INCLUDE = None

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extensions(self):
        try:
            subprocess.check_call(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DCUTLASS_ROOT=' + CUTLASS_ROOT,
            '-DCUTLASS_INCLUDE=' + CUTLASS_INCLUDE,
        ]
        
        # PyTorch 경로 추가
        if TORCH_PATH:
            cmake_args += [
                '-DTorch_DIR=' + os.path.join(TORCH_PATH, 'share', 'cmake', 'Torch'),
            ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # CMakeLists.txt가 있는 디렉토리로 sourcedir 설정
        cmake_sourcedir = PROJECT_ROOT if not ext.sourcedir else ext.sourcedir
        subprocess.check_call(['cmake', cmake_sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='cutlass-python',
    version='0.1.0',
    author='Your Name',
    description='PyTorch integration for CUTLASS examples',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    ext_modules=[CMakeExtension('cutlass_python._cutlass', sourcedir=PROJECT_ROOT)],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.19.0',
        'pybind11>=2.6.0',
    ],
)

