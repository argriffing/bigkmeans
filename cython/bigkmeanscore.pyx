"""
This is for the inner loop of the kmeans.

For compilation instructions see
http://docs.cython.org/src/reference/compilation.html
For example:
$ cython -a bigkmeanscore.pyx
$ gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
      -I/usr/include/python2.7 -o bigkmeanscore.so bigkmeanscore.c
"""

from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np
cimport cython
#from libc.math cimport log

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def update_centroids(
        np.float64_t [:, :] data_points,
        np.float64_t [:, :] centroids,
        np.int_t [:] labels,
        ):
    cdef long ndims = centroids.shape[1]
    cdef long npoints = data_points.shape[0]
    cdef long i, k
    for i in range(npoints):
        for k in range(ndims):
            centroids[labels[i], k] += data_points[i, k]
    return None

@cython.boundscheck(False)
@cython.wraparound(False)
def update_labels(
        np.float64_t [:, :] data_points,
        np.float64_t [:, :] centroids,
        np.int_t [:] labels,
        np.int_t [:] cluster_sizes,
        ):
    cdef long ncentroids = centroids.shape[0]
    cdef long ndims = centroids.shape[1]
    cdef long npoints = data_points.shape[0]
    cdef long i, j, k
    cdef long best_centroid_index
    cdef double best_centroid_dsquared
    cdef double dsquared
    cdef double delta
    for i in range(npoints):
        best_centroid_index = -1
        best_centroid_dsquared = -1
        for j in range(ncentroids):
            dsquared = 0
            for k in range(ndims):
                delta = data_points[i, k] - centroids[j, k]
                dsquared += delta * delta
            if best_centroid_index == -1 or dsquared < best_centroid_dsquared:
                best_centroid_index = j
                best_centroid_dsquared = dsquared
        labels[i] = best_centroid_index
        cluster_sizes[best_centroid_index] += 1
    return None

