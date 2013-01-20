
from StringIO import StringIO
import random
import tempfile

import numpy as np
from numpy import testing
import scipy.cluster
import h5py

import bigkmeans


def get_tmp_filename():
    f = tempfile.NamedTemporaryFile()
    name = f.name
    f.close()
    return name


class Test_BigKmeans(testing.TestCase):

    def test_medium_sized_data(self):

        # initialize the data
        M = 20000
        N = 5
        niters = 2
        nclusters = 3
        data = np.random.randn(M, N)
        indices = sorted(random.sample(xrange(M), N))
        guess = data[indices, :]

        # write an hdf file and create an associated data set
        name_hdf = get_tmp_filename()
        f_hdf = h5py.File(name_hdf)
        dset = f_hdf.create_dataset('testset', data=data)
        f_hdf.close()
        f_hdf = h5py.File(name_hdf, 'r')
        dset = f_hdf['testset']

        # write a tabular text file and re-open the file
        name_stream = get_tmp_filename()
        np.savetxt(name_stream, data)
        f_stream = open(name_stream)

        # get the scipy kmeans results
        vq_final_clust, vq_labels = scipy.cluster.vq.kmeans2(
                data, guess, niters)

        # get the bigkmeans numpy results
        np_init_clust, np_final_clust, np_labels = bigkmeans.kmeans(
                data, niters, centroids=guess)

        # get the bigkmeans hdf results
        hdf_init_clust, hdf_final_clust, hdf_labels = bigkmeans.kmeans_hdf(
                dset, niters, centroids=guess)

        # get the bigkmeans tabular text-based out-of-core results
        ooc_init_clust, ooc_final_clust, ooc_labels = bigkmeans.kmeans_ooc(
                f_stream, niters, centroids=guess)

        # check that the outputs are the same for all methods
        for final_clust in (np_final_clust, hdf_final_clust, ooc_final_clust):
            testing.assert_allclose(vq_final_clust, final_clust)
        for labels in (np_labels, hdf_labels, ooc_labels):
            testing.assert_allclose(vq_labels, labels)

        # close the hdf file and the tabular data file
        f_hdf.close()
        f_stream.close()


if __name__ == "__main__":
    testing.run_module_suite()


