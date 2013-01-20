"""
Unfortunately numpy.fromfile does not work with StringIO objects.
"""

import random

import numpy as np
from numpy import testing

import bigkmeanscore


def _lines_to_ndarray_2d(lines):
    """
    This is a hack because np.loadtxt and np.fromfile are not quite right.
    The problem with np.loadtxt is that it is too slow,
    and the problem with np.fromfile is that it does not let you
    make 2d arrays without knowing the shape beforehand.
    """
    return np.array(
            [[float(x) for x in line.split()] for line in lines],
            dtype=float)

def lloyd_update_block(
        data, curr_centroids, next_centroids, labels, cluster_sizes):
    """
    @param data: an ndarray representing a data block of a few observations
    @param curr_centroids: all current centroids
    @param next_centroids: all next centroids (to be filled)
    @param labels: data labels (to be filled)
    @param cluster_sizes: cluster sizes (to be filled)
    """
    bigkmeanscore.update_labels(data, curr_centroids, labels, cluster_sizes)
    bigkmeanscore.update_centroids(data, next_centroids, labels)

def kmeans(data, guess, niters):
    """
    This function is mostly for testing.
    The args signature is patterned after scipy.cluster.vq.kmeans2.
    @param data: an ndarray of floats with shape (M, N) with M observations
    @param guess: the initial centroids as a shape (k, N) ndarray
    @param niters: do this many Lloyd algorithm iterations
    @return: (centroids, labels)
    """
    M, N = data.shape
    nclusters = guess.shape[0]
    cluster_sizes = np.empty(nclusters, dtype=int)
    labels = np.empty(M, dtype=int)
    curr_centroids = guess.copy()
    next_centroids = guess.copy()
    for i in range(niters):

        next_centroids.fill(0)
        cluster_sizes.fill(0)
        labels.fill(-1)

        lloyd_update_block(
                data, curr_centroids, next_centroids, labels, cluster_sizes)

        if not all(cluster_sizes):
            raise Exception('empty cluster')

        next_centroids /= cluster_sizes[:, np.newaxis]

        curr_centroids, next_centroids = next_centroids, curr_centroids

    return curr_centroids, labels


def kmeans_ooc(data_stream, niters, centroids=None, nclusters=None):
    """
    If an initial cluster
    @param data_stream: a restartable stream of data
    @param niters: do this many Lloyd iterations
    @param centroids: the initial centroids as a shape (nclusters, N) ndarray
    @param nclusters: the requested number of clusters
    @return: initial_centroids, estimated_centroids, labels
    """
    if (centroids, nclusters).count(None) != 1:
        raise Exception(
                'either the initial centroids '
                'or a requested number of clusters '
                'must be provided, but not both')
    if centroids is None:
        M, N =_kmeans_ooc_scout(data_stream)
        data_stream.seek(0)
        centroids = _kmeans_ooc_init_centroids(data_stream, M, N, nclusters)
        data_stream.seek(0)
    else:
        nclusters = centroids.shape[1]
    if nclusters < 2:
        raise Exception('at least two clusters are required')
    return _kmeans_ooc_iterate(data_stream, centroids, niters)


def _kmeans_ooc_scout(data_stream):
    """
    Scout some information about the total size and dimensions of the data.
    @param data_stream: a restartable stream of data
    @return: (observation_count, ndimensions_per_observation)
    """
    M = 0
    N = None
    for line in data_stream:
        M += 1
        if N is None:
            N = len(line.split())
    return M, N


def _kmeans_ooc_init_centroids(data_stream, M, N, nclusters):
    """
    Pick some random observations out of the data stream.
    @param data_stream: a restartable stream of data
    @param M: the total number of observations
    @param N: the number of dimensions per observation
    @param nclusters: the requested number of clusters
    @return: an ndarray of shape (nclusters, N) as an initial guess
    """
    if M < 2:
        raise Exception('at least two observations are required')
    if N < 2:
        raise Exception('at least two dimensions per observation is required')
    if nclusters < 2:
        raise Exception('at least two clusters are required')
    sampled_indices = sorted(random.sample(xrange(M), nclusters))
    guess_lines = []
    for i, line in enumerate(data_stream):
        if i == sampled_indices[len(guess_lines)]:
            guess_lines.append(line)
            if len(guess_lines) == nclusters:
                break
    return _lines_to_ndarray_2d(guess_lines)


def _kmeans_ooc_iterate(data_stream, guess, niters):
    """
    @return guess, centroids, labels
    """

    if niters < 1:
        raise Exception('not enough iterations')

    # define a constant
    rows_per_block = 1000

    # init some stuff
    nclusters, N = guess.shape
    cluster_sizes = np.empty(nclusters, dtype=int)
    curr_centroids = guess.copy()
    next_centroids = guess.copy()

    for i in range(niters):

        # restart the data stream if this is not the first pass
        if i > 0:
            data_stream.seek(0)

        # these arrays will be filled as the data blocks are processed
        all_labels = []
        next_centroids.fill(0)
        cluster_sizes.fill(0)

        # process the data stream block by block
        while True:

            # read a few lines
            flat_data = np.fromfile(
                    data_stream,
                    dtype=float,
                    count=rows_per_block*N,
                    sep=' ',
                    )
            if not sum(flat_data.shape):
                break

            data = np.reshape(flat_data, (-1, N))

            M = data.shape[0]
            labels = np.empty(M, dtype=int)

            lloyd_update_block(
                    data, curr_centroids, next_centroids, labels, cluster_sizes)
            all_labels.extend(labels)

        if not all(cluster_sizes):
            raise Exception('empty cluster')

        next_centroids /= cluster_sizes[:, np.newaxis]

        curr_centroids, next_centroids = next_centroids, curr_centroids

    return guess, curr_centroids, np.array(all_labels, dtype=int)


