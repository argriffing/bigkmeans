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

import warnings
import random

import numpy as np

import lloyd

try:
    import pyvqcore
except ImportError:
    pyvqcore = None

if not pyvqcore:
    try:
        import scipy.cluster as scipy_cluster
    except ImportError:
        scipy_cluster = None

__all__ = [
        'kmeans',
        'kmeans_hdf',
        'kmeans_ooc',
        'ClusterLossError',
        'ignore_cluster_loss',
        'error_on_cluster_loss',
        'return_on_cluster_loss',
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


class ClusterLossError(Exception):
    pass

class RecoverableClusterLossError(Exception):
    pass

class ReturnClusterLossError(Exception):
    pass


def ignore_cluster_loss():
    pass

def error_on_cluster_loss():
    raise ClusterLossError('empty cluster')

def return_on_cluster_loss():
    raise ReturnClusterLossError

def retry_after_cluster_loss():
    raise RecoverableClusterLossError



##############################################################################
# These functions are trying to be like an object oriented API.

def kmeans(
        data,
        centroids=None, nclusters=None, on_cluster_loss=None,
        maxiters=None, maxrestarts=None,
        ):
    return generic_kmeans(
            data,
            np_get_shape, np_get_random_guess, np_accum,
            centroids, nclusters, on_cluster_loss,
            maxiters, maxrestarts,
            )


def kmeans_hdf(
        dset,
        centroids=None, nclusters=None, on_cluster_loss=None,
        maxiters=None, maxrestarts=None,
        ):
    return generic_kmeans(
            dset,
            hdf_get_shape, hdf_get_random_guess, hdf_accum,
            centroids, nclusters, on_cluster_loss,
            maxiters, maxrestarts,
            )

def kmeans_ooc(
        data_stream,
        centroids=None, nclusters=None, on_cluster_loss=None,
        maxiters=None, maxrestarts=None,
        ):
    return generic_kmeans(
            data_stream,
            fstream_get_shape, fstream_get_random_guess, fstream_accum,
            centroids, nclusters, on_cluster_loss,
            maxiters, maxrestarts,
            )


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
    return data_object[indices, :]

def hdf_get_random_guess(data_object, M, N, nclusters):
    indices = sorted(random.sample(xrange(M), nclusters))
    return data_object[indices, :]

def fstream_get_random_guess(data_object, M, N, nclusters):
    data_object.seek(0)
    indices = sorted(random.sample(xrange(M), nclusters))
    guess_lines = []
    for i, line in enumerate(data_object):
        if i == indices[len(guess_lines)]:
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
        data_object,
        fn_get_shape, fn_get_random_guess, fn_accum,
        centroids=None, nclusters=None, on_cluster_loss=None,
        maxiters=None, maxrestarts=None,
        ):
    """
    @param data_object: a numpy array or hdf5 dataset or open text stream
    @param fn_get_shape: a function that gets the shape of the data
    @param fn_get_random_guess: a function that randomly reinitializes clusters
    @param fn_accum: a function that customizes the core of the iteration
    @param centroids: initial centroids
    @param nclusters: number of clusters requested
    @param on_cluster_loss: call this when a cluster loss is detected
    @param maxiters: call the clustering successful after this many iterations
    @param maxrestarts: allow this many attempts to find nonempy clusters
    @return: guess_centroids, final_centroids, labels
    """
    if (maxiters is not None) and (maxiters < 1):
        raise Exception('not enough iterations')
    if (centroids, nclusters).count(None) != 1:
        raise Exception(
                'either the initial centroids '
                'or a requested number of clusters '
                'must be provided, but not both')
    M, N = fn_get_shape(data_object)
    if centroids is not None:
        nclusters = centroids.shape[0]
    if nclusters > M:
        raise Exception(
                'the number of requested clusters (%s) '
                'exceeds the number of observations (%s)' % (nclusters, M))
    if centroids is None:
        centroids = fn_get_random_guess(data_object, M, N, nclusters)

    # save the labels so that we can detect convergence
    prev_labels = None

    curr_cluster_sizes = np.ones(nclusters, dtype=int)
    next_cluster_sizes = np.zeros(nclusters, dtype=int)
    curr_centroids = centroids.copy()
    next_centroids = centroids.copy()

    restart_count = 0
    iteration_count = 0
    while True:

        # check if we are forced to stop because of the iteration cap
        if (maxiters is not None) and (iteration_count >= maxiters):
            break

        label_list = []
        next_centroids.fill(0)
        next_cluster_sizes.fill(0)

        rss = fn_accum(
                data_object, M, N, label_list,
                curr_centroids, next_centroids,
                curr_cluster_sizes, next_cluster_sizes,
                )

        # get the lost and the okay cluster indices
        lost_indices = [i for i, x in enumerate(next_cluster_sizes) if not x]
        okay_indices = [i for i, x in enumerate(next_cluster_sizes) if x]

        if any(lost_indices):
            try:
                if on_cluster_loss:
                    on_cluster_loss()
            except ReturnClusterLossError as e:
                # This is a signal to return the previous info.
                if not prev_labels:
                    raise ClusterLossError(
                            'no non-empty labeling was ever found')
                return (
                        centroids,
                        curr_centroids,
                        np.array(prev_labels, dtype=int),
                        )
            except RecoverableClusterLossError as e:
                # This is a signal to restart with a new guess.
                if maxrestarts is not None:
                    if restart_count >= maxrestarts:
                        raise Exception(
                                'hit the maxrestart cap without '
                                'finding a clustering that did not include '
                                'empty clusters')
                warnings.warn('restarting after cluster loss')
                centroids = fn_get_random_guess(data_object, M, N, nclusters)
                prev_labels = None
                curr_cluster_sizes = np.ones(nclusters, dtype=int)
                next_cluster_sizes = np.zeros(nclusters, dtype=int)
                curr_centroids = centroids.copy()
                next_centroids = centroids.copy()
                iteration_count = 0
                restart_count += 1
                continue

        # Give lost clusters their old centroids,
        # and normalize the centroids of the clusters that were not lost.
        if any(lost_indices):
            next_centroids[lost_indices, :] = curr_centroids[
                    lost_indices, :]
            next_centroids[okay_indices, :] /= next_cluster_sizes[okay_indices][
                    :, np.newaxis]
        else:
            next_centroids /= next_cluster_sizes[:, np.newaxis]

        curr_centroids, next_centroids = (
                next_centroids, curr_centroids)
        curr_cluster_sizes, next_cluster_sizes = (
                next_cluster_sizes, curr_cluster_sizes)

        # if the labels match the prev labels then we are done
        if (prev_labels is not None) and (prev_labels == label_list):
            break

        prev_labels = label_list

        iteration_count += 1

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
    @return: residue sum of squares
    """

    #FIXME: these three separate functions should be unit-tested
    #       against each other.

    if pyvqcore:
        # Try using a custom C extension if available.
        fn = lloyd.update_block_pyvqcore
    elif scipy_cluster:
        # Fall back to a scipy.cluster function if available.
        # This is slightly less vectorized, but still OK.
        fn = lloyd.update_block_scipy
    else:
        # Fall back to a pure python and numpy implementation
        # as a last resort.
        fn = lloyd.update_block_python

    return fn(
            data, curr_centroids, next_centroids, labels,
            curr_cluster_sizes, next_cluster_sizes)

