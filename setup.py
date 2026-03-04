# setup.py
import os
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

class BuildExt(build_ext):
    def build_extensions(self):
        ct = self.compiler.compiler_type
        compileOptions = []
        linkOptions = []
        
        if ct == 'msvc':
            compileOptions.append('/fopenmp')
            linkOptions.append('/fopenmp')
        else:
            compileOptions.append('-fopenmp')
            linkOptions.append('-fopenmp')
            
        for ext in self.extensions:
            if not ext.extra_compile_args:
                ext.extra_compile_args = []
            ext.extra_compile_args.extend(compileOptions)
            
            if not ext.extra_link_args:
                ext.extra_link_args = []
            ext.extra_link_args.extend(linkOptions)            
            
        super().build_extensions()

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
EIGEN3_INCLUDE_DIR = os.environ.get('EIGEN3_INCLUDE_DIR')
THREAD_POOL_INCLUDE_DIR = os.environ.get('THREAD_POOL_INCLUDE_DIR')
  
__version__ = "0.1.0"

ext_modules = [
    Pybind11Extension(
        "l0l2", 
        sources=["python/pybindwrapper.cpp"],
        include_dirs=[pybind11.get_include(), CURRENT_DIR, EIGEN3_INCLUDE_DIR, THREAD_POOL_INCLUDE_DIR],
        language='c++',
        cxx_std='latest',# Use C++23 or later
        define_macros = [('_CRT_SECURE_NO_WARNINGS', None), ('_SILENCE_ALL_CXX23_DEPRECATION_WARNINGS', None)],
    ),
]


setup(
    name='l0l2',
    version=__version__,
    author='Mamadou Thiao',
    description='l0l2 sparse modeling solver pybind11 package',
    license = 'Boost Software License - Version 1.0 - August 17th, 2003',
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    # The following ensures pybind11 is available during the build process
    setup_requires=['pybind11>=3.0.0'],
    zip_safe=False,
    python_requires=">=3.13",
    extras_require={"linearalgebra": "eigen3", "threadpool": "threadpool", "parallel": "openmp"},
)
