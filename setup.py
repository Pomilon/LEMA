from setuptools import setup, find_packages
import os

ext_modules = []
cmdclass = {}

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    ext_modules.append(
        CUDAExtension(
            name='lema.csrc._lema_cpp',
            sources=['src/lema/csrc/memory_manager.cpp'],
            extra_compile_args={'cxx': ['-O3', '-std=c++17'], 'nvcc': ['-O3', '-std=c++17']},
        ),
    )
    cmdclass['build_ext'] = BuildExtension
except Exception:
    pass  # CUDA not available — pure Python fallback

setup(
    name='lema',
    version='1.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
