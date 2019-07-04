from setuptools import setup
from Cython.Build import cythonize

setup(
    name='GF2Matrix',
    version='0.1',
    packages=['GF2Matrix'],
    package_dir={'': 'GF2Matrix'},
    url='https://github.com/EwanGilligan/GF2Matrix',
    license='MIT',
    author='Ewan Gilligan',
    author_email='eg207@st-andrews.ac.uk',
    description='Cython implementation of a matrix with entries from GF(2).',
    ext_modules=cythonize("int_matrix.pyx")
)
