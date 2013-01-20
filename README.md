bigkmeans
=========

This is a Cython-enhanced out-of-core Forgy-initialized
Lloyd's algorithm kmeans implementation for Big Data.


how to install
--------------

This is a python package and an associated script.
The install is a standard distutils-based installation
as described here -- http://docs.python.org/2/install/index.html .
It requires newish versions of the following ingredients:

 * Python 2.7+ (but not 3.x) -- http://python.org/
 * Cython -- http://cython.org/
 * Numpy -- http://www.numpy.org/

Presumably all of these dependencies exist for most operating systems.


example
-------

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

`big-data-kmeans.py --niters 2 --data data.txt --initial initial.txt`

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

`big-data-kmeans.py --niters 2 --data data.txt --initial initial.txt
	--centroids-out centroids.txt`

which should write the centroids into a file that looks like

	1.250000000000000000e+00 1.500000000000000000e+00
	3.899999999999999911e+00 5.099999999999999645e+00

If the script is giving you errors then you can try checking
its command line options using the command

`big-data-kmeans.py -h`

which should show something like


	usage: big-data-kmeans.py [-h] --niters NITERS [--nclusters NCLUSTERS] --data
				  DATA [--initial-centroids INITIAL_CENTROIDS]
				  [--labels-out LABELS_OUT]
				  [--centroids-out CENTROIDS_OUT]

	This is a script that uses the bigkmeans python module. The bigkmeans python
	module in turn uses the bigkmeanscore cython extension module. The cluster
	centroids are chosen at random from the observations.

	optional arguments:
	  -h, --help            show this help message and exit
	  --niters NITERS       do this many iterations of the Lloyd algorithm
	  --nclusters NCLUSTERS
				use this many clusters
	  --data DATA           each row of this large data file is an observation
	  --initial-centroids INITIAL_CENTROIDS
				each row of this optional file is an initial centroid
	  --labels-out LABELS_OUT
				write the labels to this file (default is stdout)
	  --centroids-out CENTROIDS_OUT
				write the centroids to this file

but it will probably be slightly different,
because I find it unlikely that I will keep this documentation
synchronized with the actual behavior of the program.
