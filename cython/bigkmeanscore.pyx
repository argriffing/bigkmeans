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

np.import_array()

#@cython.boundscheck(False)
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


#@cython.boundscheck(False)
@cython.wraparound(False)
def update_labels(
        np.float64_t [:, :] data_points,
        np.float64_t [:, :] centroids,
        np.int_t [:] labels,
        np.int_t [:] curr_cluster_sizes,
        np.int_t [:] next_cluster_sizes,
        ):
    """
    @param data_points: each row is an observation
    @param centroids: each row is a centroid
    @param labels: update these assignments
    @param curr_cluster_sizes: this lets us avoid comparisons to empty clusters
    @param next_cluster_sizes: accumulate cluster counts into this array
    @return: residual sum of squares
    """
    cdef long ncentroids = centroids.shape[0]
    cdef long ndims = centroids.shape[1]
    cdef long npoints = data_points.shape[0]
    cdef long i, j, k
    cdef long best_centroid_index
    cdef double best_centroid_dsquared
    cdef double dsquared
    cdef double delta
    cdef double rss = 0
    if curr_cluster_sizes.shape[0] != ncentroids:
        raise ValueError
    if next_cluster_sizes.shape[0] != ncentroids:
        raise ValueError
    for i in range(npoints):
        best_centroid_index = -1
        best_centroid_dsquared = -1
        for j in range(ncentroids):
            if not curr_cluster_sizes[j]:
                continue
            dsquared = 0
            for k in range(ndims):
                delta = data_points[i, k] - centroids[j, k]
                dsquared += delta * delta
            if best_centroid_index == -1 or dsquared < best_centroid_dsquared:
                best_centroid_index = j
                best_centroid_dsquared = dsquared
        rss += best_centroid_dsquared
        labels[i] = best_centroid_index
        next_cluster_sizes[best_centroid_index] += 1
    return rss

