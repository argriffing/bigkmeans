bigkmeans
=========

This is a Cython-enhanced out-of-core Forgy-initialized
Lloyd's algorithm kmeans implementation for Big Data.


why this program is so awesome
------------------------------

Theoretically it will work on data files
that contain many observations.
Other programs that try to hold all of the data in memory at once
will have a bad time if the data is huge enough.
But this program will only use a small amount of memory
even when the number of observations is large.


if you have small data then you should use something else
---------------------------------------------------------

If you only have a few data points that you want to cluster,
then using this program would be kind of dumb;
you should use R or something instead.

 * R -- http://stat.ethz.ch/R-manual/R-devel/library/stats/html/kmeans.html
 * SciPy (Python) -- http://docs.scipy.org/doc/scipy/reference/cluster.vq.html


how to install
==============

dependencies
------------

You will need newish versions of the following projects:

 * Python 2.7+ (but not 3.x) -- http://python.org/
 * Cython -- http://cython.org/
 * Numpy -- http://www.numpy.org/

These should all be available for most operating systems.

standard installation
---------------------

This is a python package and an associated script.
The install is a standard distutils-based installation
as described here -- http://docs.python.org/2/install/index.html .

install using pip
-----------------

One of several python installation helpers is called
pip -- http://www.pip-installer.org/ .
You can use this to install directly from github
using the following command.

`pip install https://github.com/argriffing/bigkmeans/zipball/master`

You might have to separately install pip itself,
and the pip command above may need to be
run using some kind of administrator privileges.

The nice thing about using pip instead of the standard installer
is that it allows you to easily uninstall the bigkmeans using the command

`pip uninstall bigkmeans`

if my kmeans clustering program is not working for you for some reason,
which frankly will probably be the case.


optional packages required for advanced usage
------------------------------------

Running kmeans on a long sequence of observations
is probably going to be I/O limited rather than CPU limited.
To reduce the I/O slowness,
you can use HDF technologies
( http://www.hdfgroup.org/ )
to store your data in a way that allows faster processing
than tabular text formats.
Support for this format is enabled if the h5py
package has been installed.

 * h5py -- http://www.h5py.org/


basic example
=============

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

`big-data-kmeans.py --maxiters 2
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

`big-data-kmeans.py --maxiters 2
	--tabular-data-file data.txt
	--initial-centroids initial.txt
	--centroids-out centroids.txt`

which should write the centroids into a file that looks like

	1.250000000000000000e+00 1.500000000000000000e+00
	3.899999999999999911e+00 5.099999999999999645e+00


troubleshooting
===============

If the script is giving you errors then you can try checking
its command line options using the command

`big-data-kmeans.py -h`

which should show something like
	
	usage: big-data-kmeans.py [-h]
				  (--nclusters NCLUSTERS | --initial-centroids INITIAL_CENTROIDS)
				  [--allow-cluster-loss | --error-on-cluster-loss | --return-on-cluster-loss | --random-restart-on-cluster-loss]
				  (--tabular-data-file TABULAR_DATA_FILE | --hdf-data-file HDF_DATA_FILE)
				  [--maxiters MAXITERS] [--maxrestarts MAXRESTARTS]
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

