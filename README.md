bigkmeans
=========

This is a Cython-enhanced out-of-core Forgy-initialized
Lloyd's algorithm kmeans implementation for Big Data.


how to install
--------------

This is a python package and an associated script.
It requires newish versions of the following ingredients:
 * Python 2.7+ (but not 3.x) -- http://python.org/
 * Cython -- http://cython.org/
 * Numpy -- http://www.numpy.org/


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
in a file called `centroids.txt`:

	1	1
	5	7



