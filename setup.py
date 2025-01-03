from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="lib.fpn.box_intersections_cpu.bbox",
        sources=["lib/fpn/box_intersections_cpu/bbox.pyx"],
        include_dirs=[np.get_include()],  # Numpy 헤더 파일 경로 포함
    ),
]

setup(
    name="egtr",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
