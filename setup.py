from setuptools import setup, find_packages

ext_modules = []
cmdclass = {}

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    ext_modules.append(
        CUDAExtension(
            name='lema._csrc._lema_cpp',
            sources=['src/lema/_csrc/memory_manager.cpp', 'src/lema/_csrc/w8a8_cuda.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '-std=c++17',
                         '-gencode=arch=compute_60,code=sm_60',
                         '-gencode=arch=compute_60,code=compute_60',
                         '-gencode=arch=compute_70,code=sm_70',
                         '-gencode=arch=compute_75,code=sm_75',
                         '-gencode=arch=compute_75,code=compute_75',
                         '-gencode=arch=compute_80,code=sm_80',
                         '-gencode=arch=compute_80,code=compute_80'],
            },
        ),
    )
    cmdclass['build_ext'] = BuildExtension
except Exception:
    pass

try:
    from torch.utils.cpp_extension import CppExtension, BuildExtension
    ext_modules.append(
        CppExtension(
            name='lema._csrc._w8a8_cpp',
            sources=['src/lema/_csrc/w8a8.cpp'],
            extra_compile_args=['-O3', '-std=c++17', '-mavx2', '-mfma'],
        ),
    )
    cmdclass['build_ext'] = BuildExtension
except Exception:
    pass

setup(
    name='lema',
    version='1.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
