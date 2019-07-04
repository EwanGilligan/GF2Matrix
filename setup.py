from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='GF2Matrix',
    version='0.2',
    packages=['GF2Matrix'],
    #package_dir={'': 'GF2Matrix'},
    url='https://github.com/EwanGilligan/GF2Matrix',
    license='MIT',
    author='Ewan Gilligan',
    author_email='eg207@st-andrews.ac.uk',
    description='Cython implementation of a matrix with entries from GF(2).',
    install_requires=['numpy', 'cython'],
    ext_modules=cythonize("GF2Matrix/int_matrix.pyx"),
    include_dirs=[numpy.get_include()]
)
