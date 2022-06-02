from setuptools import Extension, setup
import numpy as np

try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON=False
else:
    USE_CYTHON=True
    
ext=".pyx" if USE_CYTHON else ".cpp"
    
extensions=[
    Extension(
        name="mefm.model.model",
        sources=["mefm/model/model"+ext],
        include_dirs=[np.get_include()]
    ),
]

if USE_CYTHON:
    
    extensions=cythonize(
        module_list=extensions,
        compiler_directives={
            "embedsignature": True,
            "language_level": "3"
        }
    )
    
setup(
    name="mefm",
    ext_modules=extensions
)