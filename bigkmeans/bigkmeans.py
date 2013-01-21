"""
A few implementations of kmeans using a few data types.

This module has variants for three different data objects:
 * a numpy float ndarray
 * an hdf5 dataset
 * an open restartable file of data in a tabular text format
Unfortunately numpy.fromfile does not work with StringIO objects,
so this makes testing a little trickier.
If this module becomes more complicated then maybe it would be worth
reorganizing it into an object oriented API,
but for now I think that it is simple enough
that a function-oriented is better.
"""

import random

import numpy as np
from numpy import testing

import bigkmeanscore


__all__ = [
        'kmeans',
        'kmeans_hdf',
        'kmeans_ooc',
        #'ALLOW_CLUSTER_LOSS',
        #'RANDOM_RESTART_ON_CLUSTER_LOSS',
        #'ERROR_ON_CLUSTER_LOSS',
        'ignore_cluster_loss',
        'complain_bitterly_about_cluster_loss',
        'retry_after_cluster_loss',
        ]

# This constant is for buffering data.
# The Cython kmeans implementation acts on this many observations at a time
# to update the numpy arrays.
# If this number is too small, then the numpy conversion and cython call
# will be made too frequently and will cause slowness.
# If this number is too big, then the computer will try to bite off
# a chunk of memory that is too big for the RAM.
ROWS_PER_BLOCK = 8192

# define the things to do when one or more clusters are lost
#ALLOW_CLUSTER_LOSS = 'allow-cluster-loss'
#RANDOM_RESTART_ON_CLUSTER_LOSS = 'random-restart-on-cluster-loss'
#ERROR_ON_CLUSTER_LOSS = 'error-on-cluster-loss'


class ClusterLossError(Exception):
    pass

class RecoverableClusterLossError(Exception):
    pass


def ignore_cluster_loss():
    pass

def complain_bitterly_about_cluster_loss():
    raise ClusterLossError('empty cluster')

def retry_after_cluster_loss():
    raise RecoverableClusterLossError



##############################################################################
# These functions are trying to be like an object oriented API.

def kmeans(
        data, niters,
        centroids=None, nclusters=None, on_cluster_loss=ignore_cluster_loss):
    return generic_kmeans(
            data, niters,
            np_get_shape, np_get_random_guess, np_accum,
            centroids, nclusters, on_cluster_loss)

def kmeans_hdf(
        dset, niters,
        centroids=None, nclusters=None, on_cluster_loss=ignore_cluster_loss):
    return generic_kmeans(
            dset, niters,
            hdf_get_shape, hdf_get_random_guess, hdf_accum,
            centroids, nclusters, on_cluster_loss)

def kmeans_ooc(
        data_stream, niters,
        centroids=None, nclusters=None, on_cluster_loss=ignore_cluster_loss):
    return generic_kmeans(
            data_stream, niters,
            fstream_get_shape, fstream_get_random_guess, fstream_accum,
            centroids, nclusters, on_cluster_loss)


##############################################################################
# For each data object type, get the number and size of observations.

def np_get_shape(data_object):
    return data_object.shape

def hdf_get_shape(data_object):
    return data_object.shape

def fstream_get_shape(data_object):
    data_object.seek(0)
    M = 0
    N = None
    for line in data_object:
        M += 1
        if N is None:
            N = len(line.split())
    return M, N


##############################################################################
# For each data object type, get a random centroid initialization.

def np_get_random_guess(data_object, M, N, nclusters):
    indices = sorted(random.sample(xrange(M), nclusters))
    return indices[observation_indices, :]

def hdf_get_random_guess(data_object, M, N, nclusters):
    indices = sorted(random.sample(xrange(M), nclusters))
    return indices[observation_indices, :]

def fstream_get_random_guess(data_object, M, N, nclusters):
    data_object.seek(0)
    indices = sorted(random.sample(xrange(M), nclusters))
    guess_lines = []
    for i, line in enumerate(data_object):
        if i == sampled_indices[len(guess_lines)]:
            guess_lines.append(line)
            if len(guess_lines) == nclusters:
                break
    arr = [[float(x) for x in line.split()] for line in guess_lines]
    return np.array(arr, dtype=float)


##############################################################################
# For each data object type, accumulate info over a single kmeans iteration.
# Return the residue sum of squares.

def np_accum(
        data_object, M, N, label_list,
        curr_centroids, next_centroids,
        curr_cluster_sizes, next_cluster_sizes,
        ):
    labels = np.empty(M, dtype=int)
    rss = lloyd_update_block(
            data_object, curr_centroids, next_centroids, labels,
            curr_cluster_sizes, next_cluster_sizes)
    label_list.extend(labels)
    return rss

def hdf_accum(
        data_object, M, N, label_list,
        curr_centroids, next_centroids,
        curr_cluster_sizes, next_cluster_sizes,
        ):
    # process the data stream block by block
    block_index = 0
    rss = 0
    while True:
        start = block_index * ROWS_PER_BLOCK
        stop = (block_index+1) * ROWS_PER_BLOCK
        if start >= M:
            break
        data = data_object[start : stop]
        M_block = data.shape[0]
        labels = np.empty(M_block, dtype=int)
        rss += lloyd_update_block(
                data, curr_centroids, next_centroids, labels,
                curr_cluster_sizes, next_cluster_sizes)
        label_list.extend(labels)
        block_index += 1
    return rss

def fstream_accum(
        data_object, M, N, label_list,
        curr_centroids, next_centroids,
        curr_cluster_sizes, next_cluster_sizes,
        ):
    data_object.seek(0)
    rss = 0
    while True:
        flat_data = np.fromfile(
                data_object,
                dtype=float,
                count=ROWS_PER_BLOCK*N,
                sep=' ',
                )
        if not sum(flat_data.shape):
            break
        data = np.reshape(flat_data, (-1, N))
        M_block = data.shape[0]
        labels = np.empty(M_block, dtype=int)
        rss += lloyd_update_block(
                data, curr_centroids, next_centroids, labels,
                curr_cluster_sizes, next_cluster_sizes)
        label_list.extend(labels)
    return rss


##############################################################################
# Define a generic kmeans that uses functions specific to the data type.
# This could be turned into an object oriented thing later.

def generic_kmeans(
        data_object, niters,
        fn_get_shape, fn_get_random_guess, fn_accum,
        centroids=None, nclusters=None, on_cluster_loss=ignore_cluster_loss):
    """
    @param data_object: a numpy array or hdf5 dataset or open text stream
    @param fn_get_shape: a function that gets the shape of the data
    @param fn_get_random_guess: a function that randomly reinitializes clusters
    @param fn_accum: a function that customizes the core of the iteration
    @param centroids: initial centroids
    @param nclusters: number of clusters requested
    @param on_cluster_loss: call this when a cluster loss is detected
    """
    if niters < 1:
        raise Exception('not enough iterations')
    if (centroids, nclusters).count(None) != 1:
        raise Exception(
                'either the initial centroids '
                'or a requested number of clusters '
                'must be provided, but not both')
    M, N = fn_get_shape(data_object)
    if centroids is None:
        centroids = fn_get_random_guess(data_object, M, N, nclusters)
    else:
        nclusters = centroids.shape[0]

    #TODO: begin loop for cluster-loss restarting

    curr_cluster_sizes = np.ones(nclusters, dtype=int)
    next_cluster_sizes = np.zeros(nclusters, dtype=int)
    curr_centroids = centroids.copy()
    next_centroids = centroids.copy()

    for i in range(niters):

        label_list = []
        next_centroids.fill(0)
        next_cluster_sizes.fill(0)

        rss = fn_accum(
                data_object, M, N, label_list,
                curr_centroids, next_centroids,
                curr_cluster_sizes, next_cluster_sizes,
                )

        if all(next_cluster_sizes):
            next_centroids /= next_cluster_sizes[:, np.newaxis]
        else:
            on_cluster_loss()
            for j in range(nclusters):
                if next_cluster_sizes[j]:
                    next_centroids[j] /= next_cluster_sizes[j]

        curr_centroids, next_centroids = (
                next_centroids, curr_centroids)
        curr_cluster_sizes, next_cluster_sizes = (
                next_cluster_sizes, curr_cluster_sizes)

    #TODO: end loop for cluster-loss restarting

    return centroids, curr_centroids, np.array(label_list, dtype=int)



def lloyd_update_block(
        data, curr_centroids, next_centroids, labels,
        curr_cluster_sizes, next_cluster_sizes):
    """
    @param data: an ndarray representing a data block of a few observations
    @param curr_centroids: all current centroids
    @param next_centroids: all next centroids (to be filled)
    @param labels: data labels (to be filled)
    @param curr_cluster_sizes: this is used only as an empty cluster mask
    @param next_cluster_sizes: cluster sizes (to be filled)
    """
    rss = bigkmeanscore.update_labels(
            data, curr_centroids, labels,
            curr_cluster_sizes, next_cluster_sizes)
    bigkmeanscore.update_centroids(data, next_centroids, labels)
    return rss

