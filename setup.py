from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [Extension("GF2Matrix.int_matrix",
                        ["GF2Matrix/int_matrix.pyx"],
                        include_dirs=[numpy.get_include()]
                        )
              ]

setup(
    name='GF2Matrix-eg207',
    version='0.3',
    packages=find_packages(),
    #package_dir={'': 'GF2Matrix'},
    url='https://github.com/EwanGilligan/GF2Matrix',
    license='MIT',
    author='Ewan Gilligan',
    author_email='eg207@st-andrews.ac.uk',
    description='Cython implementation of a matrix with entries from GF(2).',
    install_requires=['numpy', 'cython'],
    ext_modules=cythonize(extensions),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
