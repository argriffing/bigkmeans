"""
Three implementations of a Lloyd algorithm iteration on a block of observations.

The first implementation uses a custom C-extension.
The second implementation uses a scipy.clustering function.
The third implementation uses a combination of Numpy and Python.
These implementations should be mostly equivalent,
but they may differ slightly when clusters are lost.
For example, I think that in some of these implementations
lost centroids could be resurrected while this would not be possible
in other implementations.
But I do not have any example of this happening.
"""

import numpy as np

#FIXME: import lazily?

try:
    import pyvqcore
except ImportError:
    pyvqcore = None

try:
    import scipy.cluster as scipy_cluster
except ImportError:
    scipy_cluster = None

__all__ = [
        'update_block_pyvqcore',
        'update_block_scipy',
        'update_block_python',
        ]

g_doc = """\
@param data: an ndarray representing a data block of a few observations
@param curr_centroids: all current centroids
@param next_centroids: all next centroids (to be filled)
@param labels: data labels (to be filled)
@param curr_cluster_sizes: this is used only as an empty cluster mask
@param next_cluster_sizes: cluster sizes (to be filled)
@return: residue sum of squares
"""

def update_block_pyvqcore(
        data, curr_centroids, next_centroids, labels,
        curr_cluster_sizes, next_cluster_sizes):

    # do everything in the C-extension
    rss = pyvqcore.update_labels(
            data, curr_centroids, labels,
            curr_cluster_sizes, next_cluster_sizes)
    pyvqcore.update_centroids(data, next_centroids, labels)
    return rss

update_block_pyvqcore.__doc__ = g_doc


def update_block_scipy(
        data, curr_centroids, next_centroids, labels,
        curr_cluster_sizes, next_cluster_sizes):

    # get the codebook index array and a distance array
    index_arr, distortion_arr = scipy_cluster.vq.vq(data, curr_centroids)
    rss = sum(np.square(distortion_arr))

    # update the labels and the next cluster sizes
    labels[...] = index_arr
    for cluster_index in labels:
        next_cluster_sizes[cluster_index] += 1

    # update the next centroids
    for i, row in enumerate(data):
        next_centroids[labels[i]] += row

    return rss

update_block_scipy.__doc__ = g_doc


def update_block_python(
        data, curr_centroids, next_centroids, labels,
        curr_cluster_sizes, next_cluster_sizes):

    # For each data point get the closest centroid.
    # Also keep track of the sum of squared errors
    # and the number of times each centroid was closest.
    # And update the next centroids.
    rss = 0
    for i, row in enumerate(data):
        ds = np.sum(np.square(curr_centroids - row), axis=1)
        d, label = min((d, j) for j, d in enumerate(ds))
        labels[i] = label
        next_cluster_sizes[label] += 1
        next_centroids[label] += row
        rss += d
    return rss

update_block_python.__doc__ = g_doc

