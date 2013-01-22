#! /usr/bin/python

"""
This is a script that uses the bigkmeans python module.

The bigkmeans python module in turn
uses the bigkmeanscore cython extension module.
The cluster centroids are chosen at random from the observations.
"""

import sys
import argparse

import numpy as np

import bigkmeans

try:
    import h5py
except ImportError:
    h5py = None



def pos_int(x):
    x = int(x)
    if x < 1:
        raise argparse.ArgumentTypeError('value must be a positive integer')
    return x

def nonneg_int(x):
    x = int(x)
    if x < 0:
        raise argparse.ArgumentTypeError('value must be a non-negative integer')
    return x


def main(args):

    # Optionally read the initial centroids.
    guess = None
    if args.initial_centroids:
        guess = np.loadtxt(args.initial_centroids, dtype=float, ndmin=2)

    # Optionally specify an inner loop implementation choice.
    fn_block_update = None
    if args.inner_loop:
        inner_loop_dict = {
                'pyvqcore' : bigkmeans.lloyd.update_block_pyvqcore,
                'scipy' : bigkmeans.lloyd.update_block_scipy,
                'python' : bigkmeans.lloyd.update_block_python,
                }
        fn_block_update = inner_loop_dict[fn_block_update]

    # Open the data file and do the kmeans clustering.
    # Note that we deliberately disallow using stdin
    # because we require that the stream can be restarted
    # so that we can do one pass through the open file per iteration.
    if args.tabular_data_file:
        with open(args.tabular_data_file) as data_stream:
            guess, centroids, labels = bigkmeans.kmeans_ooc(
                    data_stream,
                    centroids=guess,
                    nclusters=args.nclusters,
                    on_cluster_loss=args.on_cluster_loss,
                    maxiters=args.maxiters,
                    maxrestarts=args.maxrestarts,
                    fn_block_update=fn_block_update,
                    verbose=args.verbose,
                    )
    elif args.hdf_data_file:
        if not h5py:
            raise ImportError(
                    'sorry I cannot deal with hdf5 data files '
                    'unless the python package h5py is installed')
        if not args.hdf_dataset_name:
            raise Exception(
                    'If the data is in hdf format '
                    'then an hdf dataset name (--hdf-dataset-name) '
                    'must be specified '
                    'in addition to the name of the hdf file.  '
                    'If you do not know the dataset name, '
                    'then you can try to use the program called hdfview '
                    'to search for your dataset within your hdf file.')
        f = h5py.File(args.hdf_data_file, 'r')
        dset = f[args.hdf_dataset_name]
        guess, centroids, labels = bigkmeans.kmeans_hdf(
                dset,
                centroids=guess,
                nclusters=args.nclusters,
                on_cluster_loss=args.on_cluster_loss,
                maxiters=args.maxiters,
                maxrestarts=args.maxrestarts,
                fn_block_update=fn_block_update,
                verbose=args.verbose,
                )
        f.close()

    # write the labels to stdout or to a user-specified file
    if args.labels_out == '-':
        np.savetxt(sys.stdout, labels, '%d')
    elif args.labels_out:
        np.savetxt(args.labels_out, labels, '%d')

    # optionally write the centroids
    if args.centroids_out == '-':
        np.savetxt(sys.stdout, centroids)
    elif args.centroids_out:
        np.savetxt(args.centroids_out, centroids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    # initialize the cluster centroids
    centroid_init = parser.add_mutually_exclusive_group(required=True)
    centroid_init.add_argument('--nclusters', type=pos_int,
            help='use this many clusters')
    centroid_init.add_argument('--initial-centroids',
            help='each row of this optional file is an initial centroid')

    # how do we react to cluster loss
    cluster_loss = parser.add_mutually_exclusive_group()
    cluster_loss.add_argument(
            '--allow-cluster-loss',
            action='store_const',
            dest='on_cluster_loss',
            const=bigkmeans.ignore_cluster_loss,
            help='use this flag if you can tolerate some empty clusters')
    cluster_loss.add_argument(
            '--error-on-cluster-loss',
            action='store_const',
            dest='on_cluster_loss',
            const=bigkmeans.error_on_cluster_loss,
            help='cluster loss raises an error')
    cluster_loss.add_argument(
            '--return-on-cluster-loss',
            action='store_const',
            dest='on_cluster_loss',
            const=bigkmeans.return_on_cluster_loss,
            help='cluster loss immediately returns the previous clustering')
    cluster_loss.add_argument(
            '--random-restart-on-cluster-loss',
            action='store_const',
            dest='on_cluster_loss',
            const=bigkmeans.retry_after_cluster_loss,
            help='restart with random centroids after a cluster loss')

    # define the data file
    data_defn = parser.add_mutually_exclusive_group(required=True)
    data_defn.add_argument('--tabular-data-file',
            help='each row of this large tabular text file is an observation')
    data_defn.add_argument('--hdf-data-file',
            help='each row of a dataset in this hdf file is an observation')

    parser.add_argument('--maxiters', type=pos_int,
            help='say that the kmeans has converged after this many iterations')
    parser.add_argument('--maxrestarts', type=nonneg_int,
            help='allow this many random restarts to avoid cluster loss')

    parser.add_argument('--verbose', action='store_true',
            help='spam more')
    parser.add_argument(
            '--inner-loop',
            choices=['pyvqcore', 'scipy', 'python'],
            help='explicitly specify a kmeans inner loop implementation')
    parser.add_argument('--hdf-dataset-name',
            help='specify the name of the dataset within the hdf data file')
    parser.add_argument('--labels-out', default='-',
            help='write the labels to this file (default is stdout)')
    parser.add_argument('--centroids-out',
            help='write the centroids to this file')
    main(parser.parse_args())



