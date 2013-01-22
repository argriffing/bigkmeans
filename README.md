bigkmeans
=========

An implementation of Lloyd's kmeans clustering algorithm
for data sets that contain many points.


when to use something else
--------------------------

If you can hold all of the data in RAM
you should use R or something instead.

 * R -- http://stat.ethz.ch/R-manual/R-devel/library/stats/html/kmeans.html
 * SciPy (Python) -- http://docs.scipy.org/doc/scipy/reference/cluster.vq.html
 * MATLAB -- http://www.mathworks.com/help/stats/kmeans.html

But if your computer chokes when you tell it to load your data,
then this python package and its associated scripts might help you.


requirements
------------

 * [Python](http://python.org/) 2.7+ (but not 3.x)
 * [NumPy](http://www.numpy.org/)


optional python packages
------------------------

To reduce CPU slowness, install at least one of the following:
 * [pyvqcore](https://github.com/argriffing/pyvqcore)
 * [SciPy](http://www.scipy.org/)
 * [EPD](http://www.enthought.com/products/epd.php)

To reduce I/O slowness, use [hdf5](http://www.hdfgroup.org/HDF5/)
data files and install this python package:
 * [h5py](http://www.h5py.org/)


standard installation
---------------------

You can install bigkmeans using the standard distutils installation procedure
for python packages with setup.py scripts,
as explained [here](http://docs.python.org/2/install/index.html).


install using pip
-----------------

One of several Python package installation helpers is called
[pip](http://www.pip-installer.org/).
You can use this to install directly from github using the command

`$ pip install --user https://github.com/argriffing/bigkmeans/zipball/master`

which can be reverted by

`$ pip uninstall bigkmeans`


testing the installation
------------------------

To test the installation of the python package, try running the command

`$ python -c "import bigkmeans; bigkmeans.test()"`

on your command line,
where the `$` is my notation for a shell prompt rather than
something that you are supposed to type.
Your command line prompt might look different.
Also your command to run Python might need to be different;
for example you might need to use something like
`Python27.exe` or something else depending on your environment.
To test the installation of the script, try running the command

`$ big-data-kmeans.py -h`

on your command line.
This might work if you've installed the package
using administrative privileges.
Otherwise you might try a command like

`$ python ~/.local/bin/big-data-kmeans.py -h`

which might work if you've installed the package using the pip `--user` flag.


basic example
-------------

The web page
http://mnemstudio.org/clustering-k-means-example-1.htm
has a nice example of kmeans clustering.

First make a data file called `data.txt`
where each row defines a 2d observation.
    
	1	1
	1.5	2
	3	4
	5	7
	3.5	5
	4.5	5
	3.5	4.5

Continuing to follow the example on that web page,
we make a couple of initial cluster centroids
in a file called `initial.txt`:

	1	1
	5	7

Now that these two files have been created,
you can analyze the observations using the command

`$ big-data-kmeans.py --maxiters 2
	--tabular-data-file data.txt --initial-centroids initial.txt`

which should show the cluster assignments

	0
	0
	1
	1
	1
	1
	1

If you want the script to also show the centroids,
you can specify an output file using the command

`$ big-data-kmeans.py --maxiters 2
	--tabular-data-file data.txt
	--initial-centroids initial.txt
	--centroids-out centroids.txt`

which should write the centroids into a file that looks like

	1.250000000000000000e+00 1.500000000000000000e+00
	3.899999999999999911e+00 5.099999999999999645e+00


troubleshooting
---------------

If the script is giving you errors then you can try checking
its command line options using the command

`$ big-data-kmeans.py -h`

which should show something like
	
    usage: big-data-kmeans.py [-h]
                              (--nclusters NCLUSTERS | --initial-centroids INITIAL_CENTROIDS)
                              [--allow-cluster-loss | --error-on-cluster-loss | --return-on-cluster-loss | --random-restart-on-cluster-loss]
                              (--tabular-data-file TABULAR_DATA_FILE | --hdf-data-file HDF_DATA_FILE)
                              [--maxiters MAXITERS] [--maxrestarts MAXRESTARTS]
                              [--verbose] [--inner-loop {pyvqcore,scipy,python}]
                              [--hdf-dataset-name HDF_DATASET_NAME]
                              [--labels-out LABELS_OUT]
                              [--centroids-out CENTROIDS_OUT]

    This is a script that uses the bigkmeans python module. The bigkmeans python
    module in turn uses the bigkmeanscore cython extension module. The cluster
    centroids are chosen at random from the observations.

    optional arguments:
      -h, --help            show this help message and exit
      --nclusters NCLUSTERS
                            use this many clusters
      --initial-centroids INITIAL_CENTROIDS
                            each row of this optional file is an initial centroid
      --allow-cluster-loss  use this flag if you can tolerate some empty clusters
      --error-on-cluster-loss
                            cluster loss raises an error
      --return-on-cluster-loss
                            cluster loss immediately returns the previous
                            clustering
      --random-restart-on-cluster-loss
                            restart with random centroids after a cluster loss
      --tabular-data-file TABULAR_DATA_FILE
                            each row of this large tabular text file is an
                            observation
      --hdf-data-file HDF_DATA_FILE
                            each row of a dataset in this hdf file is an
                            observation
      --maxiters MAXITERS   say that the kmeans has converged after this many
                            iterations
      --maxrestarts MAXRESTARTS
                            allow this many random restarts to avoid cluster loss
      --verbose             spam more
      --inner-loop {pyvqcore,scipy,python}
                            explicitly specify a kmeans inner loop implementation
      --hdf-dataset-name HDF_DATASET_NAME
                            specify the name of the dataset within the hdf data
                            file
      --labels-out LABELS_OUT
                            write the labels to this file (default is stdout)
      --centroids-out CENTROIDS_OUT
                            write the centroids to this file

but it will probably be slightly different,
because I find it unlikely that I will keep this documentation
synchronized with the actual behavior of the program.
