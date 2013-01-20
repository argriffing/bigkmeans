
from StringIO import StringIO
import random

import numpy as np
from numpy import testing

import scipy.cluster

import bigkmeans

class Test_BigKmeans(testing.TestCase):

    def test_scipy(self):

        # init some stuff
        M = 20
        N = 5
        nclusters = 3
        niters = 2
        data = np.random.randn(M, N)

        # do the test
        guess = np.array([random.choice(data) for i in range(nclusters)])
        bk_centroids, bk_labels = bigkmeans.kmeans(
                data, guess, niters)
        print 'bigkmeans guess, centroids and labels:'
        print guess
        print bk_centroids
        print bk_labels
        print
        vq_centroids, vq_labels = scipy.cluster.vq.kmeans2(
                data, guess, niters)
        print 'scipy.cluster.vq.kmeans2 guess, centroids and labels:'
        print guess
        print vq_centroids
        print vq_labels
        print
        testing.assert_allclose(bk_centroids, vq_centroids)
        testing.assert_allclose(bk_labels, vq_labels)


    def test_kmeans_ooc(self):

        # init some stuff
        M = 200000
        N = 50
        nclusters = 100
        niters = 5
        data = np.random.randn(M, N)

        # Make a temporary file.
        # This is necessary because np.fromfile does not work with StringIO.
        # FIXME: use a file that is actually temporary
        tmp_filename = 'testing.kmeans.txt'
        np.savetxt(tmp_filename, data)

        # Do kmeans using an open file.
        with open(tmp_filename) as data_stream:
            ooc_guess, ooc_centroids, ooc_labels = bigkmeans.kmeans_ooc(
                    data_stream, niters, nclusters=nclusters)
        print 'bigkmeans ooc guess, centroids and labels:'
        print ooc_guess
        print ooc_centroids
        print ooc_labels
        print

        # Compare to scipy kmeans directly using the data in memory.
        vq_centroids, vq_labels = scipy.cluster.vq.kmeans2(
                data, ooc_guess, niters)
        print 'scipy.cluster.vq.kmeans2 guess, centroids and labels:'
        print ooc_guess
        print vq_centroids
        print vq_labels
        print
        testing.assert_allclose(ooc_centroids, vq_centroids)
        testing.assert_allclose(ooc_labels, vq_labels)



if __name__ == "__main__":
    testing.run_module_suite()

