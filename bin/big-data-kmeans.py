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
    with open(args.data) as data_stream:
        guess, centroids, labels = bigkmeans.kmeans_ooc(
                data_stream,
                args.niters,
                centroids=guess,
                nclusters=args.nclusters,
                )

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
    parser.add_argument('--niters', type=pos_int, required=True,
            help='do this many iterations of the Lloyd algorithm')
    parser.add_argument('--nclusters', type=pos_int,
            help='use this many clusters')
    parser.add_argument('--data', required=True,
            help='each row of this large data file is an observation')
    parser.add_argument('--initial-centroids',
            help='each row of this optional file is an initial centroid')
    parser.add_argument('--labels-out', default='-',
            help='write the labels to this file (default is stdout)')
    parser.add_argument('--centroids-out',
            help='write the centroids to this file')
    main(parser.parse_args())

