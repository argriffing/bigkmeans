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


def main(args):

    # Optionally read the initial centroids.
    guess = None
    if args.initial_centroids:
        guess = np.loadtxt(args.initial_centroids, dtype=float, ndmin=2)

    # Open the data file and do the kmeans clustering.
    # Note that we deliberately disallow using stdin
    # because we require that the stream can be restarted
    # so that we can do one pass through the open file per iteration.
    if args.tabular_data_file:
        with open(args.tabular_data_file) as data_stream:
            guess, centroids, labels = bigkmeans.kmeans_ooc(
                    data_stream,
                    args.niters,
                    centroids=guess,
                    nclusters=args.nclusters,
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
                args.niters,
                centroids=guess,
                nclusters=args.nclusters,
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

    # define the data file
    data_defn = parser.add_mutually_exclusive_group(required=True)
    data_defn.add_argument('--tabular-data-file',
            help='each row of this large tabular text file is an observation')
    data_defn.add_argument('--hdf-data-file',
            help='each row of a dataset in this hdf file is an observation')

    parser.add_argument('--niters', type=pos_int, required=True,
            help='do this many iterations of the Lloyd algorithm')

    parser.add_argument('--hdf-dataset-name',
            help='specify the name of the dataset within the hdf data file')
    parser.add_argument('--labels-out', default='-',
            help='write the labels to this file (default is stdout)')
    parser.add_argument('--centroids-out',
            help='write the centroids to this file')
    main(parser.parse_args())

