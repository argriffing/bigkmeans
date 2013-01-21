"""
The core algorithm has been deferred to pyvqcore or scipy.clustering.

So this package no longer includes extension modules.
"""

from distutils.core import setup

setup(
        name = 'bigkmeans',
        version = '0.1',
        packages=['bigkmeans'],
        scripts = ['bin/big-data-kmeans.py'],
        )
