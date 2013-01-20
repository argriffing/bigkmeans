
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

bigkmeanscore_ext = Extension('bigkmeanscore', ['bigkmeanscore.pyx'])

setup(
        name = 'bigkmeans',
        version = '0.1',
        py_modules = ['bigkmeans'],
        cmdclass = { 'build_ext' : build_ext },
        ext_modules = [bigkmeanscore_ext],
        )
