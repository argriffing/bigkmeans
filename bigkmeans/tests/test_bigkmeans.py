
from StringIO import StringIO
import random
import tempfile

import numpy as np
from numpy import testing
import scipy.cluster
import h5py

import bigkmeans


#http://www.math.uah.edu/stat/data/Fisher.txt
g_fisher = """\
Type    PW  PL  SW  SL
0   2   14  33  50
1   24  56  31  67
1   23  51  31  69
0   2   10  36  46
1   20  52  30  65
1   19  51  27  58
2   13  45  28  57
2   16  47  33  63
1   17  45  25  49
2   14  47  32  70
0   2   16  31  48
1   19  50  25  63
0   1   14  36  49
0   2   13  32  44
2   12  40  26  58
1   18  49  27  63
2   10  33  23  50
0   2   16  38  51
0   2   16  30  50
1   21  56  28  64
0   4   19  38  51
0   2   14  30  49
2   10  41  27  58
2   15  45  29  60
0   2   14  36  50
1   19  51  27  58
0   4   15  34  54
1   18  55  31  64
2   10  33  24  49
0   2   14  42  55
1   15  50  22  60
2   14  39  27  52
0   2   14  29  44
2   12  39  27  58
1   23  57  32  69
2   15  42  30  59
1   20  49  28  56
1   18  58  25  67
2   13  44  23  63
2   15  49  25  63
2   11  30  25  51
1   21  54  31  69
1   25  61  36  72
2   13  36  29  56
1   21  55  30  68
0   1   14  30  48
0   3   17  38  57
2   14  44  30  66
0   4   15  37  51
2   17  50  30  67
1   22  56  28  64
1   15  51  28  63
2   15  45  22  62
2   14  46  30  61
2   11  39  25  56
1   23  59  32  68
1   23  54  34  62
1   25  57  33  67
0   2   13  35  55
2   15  45  32  64
1   18  51  30  59
1   23  53  32  64
2   15  45  30  54
1   21  57  33  67
0   2   13  30  44
0   2   16  32  47
1   18  60  32  72
1   18  49  30  61
0   2   12  32  50
0   1   11  30  43
2   14  44  31  67
0   2   14  35  51
0   4   16  34  50
2   10  35  26  57
1   23  61  30  77
2   13  42  26  57
0   1   15  41  52
1   18  48  30  60
2   13  42  27  56
0   2   15  31  49
0   4   17  39  54
2   16  45  34  60
2   10  35  20  50
0   2   13  32  47
2   13  54  29  62
0   2   15  34  51
2   10  50  22  60
0   1   15  31  49
0   2   15  37  54
2   12  47  28  61
2   13  41  28  57
0   4   13  39  54
1   20  51  32  65
2   15  49  31  69
2   13  40  25  55
0   3   13  23  45
0   3   15  38  51
2   14  48  28  68
0   2   15  35  52
1   25  60  33  63
2   15  46  28  65
0   3   14  34  46
2   18  48  32  59
2   16  51  27  60
1   18  55  30  65
0   5   17  33  51
1   22  67  38  77
1   21  66  30  76
1   13  52  30  67
2   13  40  28  61
2   11  38  24  55
0   2   14  34  52
1   20  64  38  79
0   6   16  35  50
1   20  67  28  77
2   12  44  26  55
0   3   14  30  48
0   2   19  34  48
1   14  56  26  61
0   2   12  40  58
1   18  48  28  62
2   15  45  30  56
0   2   14  32  46
0   4   15  44  57
1   24  56  34  63
1   16  58  30  72
1   21  59  30  71
1   18  56  29  63
2   12  42  30  57
1   23  69  26  77
2   13  56  29  66
0   2   15  34  52
2   10  37  24  55
0   2   15  31  46
1   19  61  28  74
0   3   13  35  50
1   18  63  29  73
2   15  47  31  67
2   13  41  30  56
2   13  43  29  64
1   22  58  30  65
0   3   14  35  51
2   14  47  29  61
1   19  53  27  64
0   2   16  34  48
1   20  50  25  57
2   13  40  23  55
0   2   17  34  54
1   24  51  28  58
0   2   15  37  53
"""


def get_tmp_filename():
    f = tempfile.NamedTemporaryFile()
    name = f.name
    f.close()
    return name


class Test_BigKmeans(testing.TestCase):

    def helper(self, data, niters, guess=None, nclusters=None):

        # if no guess has been provided then we make a guess
        if guess is None:
            M, N = data.shape
            indices = sorted(random.sample(xrange(M), nclusters))
            guess = data[indices, :]
            #print 'random guess:'
            #print guess
            #print

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

    def test_random_medium_sized_data(self):

        # initialize the data
        M = 20000
        N = 5
        niters = 2
        nclusters = 3
        data = np.random.randn(M, N)
        self.helper(data, niters, nclusters=nclusters)

    def test_mnemstudio_example(self):

        # define data and the initial guess for this example
        data = np.array([
            [1, 1],
            [1.5, 2],
            [3, 4],
            [5, 7],
            [3.5, 5],
            [4.5, 5],
            [3.5, 4.5],
            ], dtype=float)
        guess = np.array([
            [1, 1],
            [5, 7],
            ], dtype=float)

        # check the labels after the first iteration
        expected_labels_iters_1 = np.array([0, 0, 0, 1, 1, 1, 1], dtype=int)
        clust, labels = scipy.cluster.vq.kmeans2(data, guess, 1)
        testing.assert_array_equal(labels, expected_labels_iters_1)

        # check the labels after the second iteration
        expected_labels_iters_2 = np.array([0, 0, 1, 1, 1, 1, 1], dtype=int)
        clust, labels = scipy.cluster.vq.kmeans2(data, guess, 2)
        testing.assert_array_equal(labels, expected_labels_iters_2)

        # check that all methods agree for multiple iterations
        for niters in range(1, 10):
            self.helper(data, niters, guess=guess)

    def test_fisher_iris_data_random_guess(self):

        # define the data
        data = np.loadtxt(
                StringIO(g_fisher),
                dtype=float,
                skiprows=1,
                usecols=(1,2,3,4),
                )

        # Do some unsupervised clustering on this data set,
        # using more than the three putative clusters.
        niters = 10
        nclusters = 20
        self.helper(data, niters, nclusters=nclusters)

    def test_fisher_iris_empty_cluster_guess(self):

        # define the data
        data = np.loadtxt(
                StringIO(g_fisher),
                dtype=float,
                skiprows=1,
                usecols=(1,2,3,4),
                )

        # this guess of initial centroids is known to lead to cluster loss
        guess = np.array([
            [  2.,  10.,  36.,  46.],
            [ 19.,  51.,  27.,  58.],
            [ 14.,  47.,  32.,  70.],
            [ 19.,  51.,  27.,  58.],
            [  2.,  14.,  42.,  55.],
            [ 14.,  39.,  27.,  52.],
            [  1.,  14.,  30.,  48.],
            [  2.,  13.,  35.,  55.],
            [ 18.,  51.,  30.,  59.],
            [  2.,  15.,  34.,  51.],
            [  2.,  15.,  37.,  54.],
            [  2.,  15.,  35.,  52.],
            [ 15.,  46.,  28.,  65.],
            [  2.,  19.,  34.,  48.],
            [ 18.,  48.,  28.,  62.],
            [ 18.,  56.,  29.,  63.],
            [ 12.,  42.,  30.,  57.],
            [ 23.,  69.,  26.,  77.],
            [  2.,  15.,  31.,  46.],
            [  2.,  17.,  34.,  54.],
            ], dtype=float)

        # Do some unsupervised clustering on this data set,
        # using more than the three putative clusters.
        niters = 10
        self.helper(data, niters, guess=guess)




if __name__ == "__main__":
    testing.run_module_suite()


