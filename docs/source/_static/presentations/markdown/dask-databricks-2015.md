`dask`
------

*Matthew Rocklin*

Continuum Analytics


### Dask is a Python library for out-of-core, parallel processing

<hr>

### mostly for a single node


`dask`
------

*  High level collections + Blocked algorithms
    *  `dask.array = numpy + threading`
    *  `dask.dataframe = pandas + threading`
    *  `dask.bag = map + ... + multiprocessing`
    *  ...
*  Dynamic task scheduler
    *  Arbitrary graphs
    *  Removes intermediates
    *  Millisecond overhead per task
*  Spark comparison
    *  Mostly single node
    *  Friendly to Python users
    *  Task graph only, no high level RDD


*  High level collections build graphs
*  Low level schedulers execute graphs

<img src="images/collections-schedulers2.png"
     width="100%">

*  Neither side needs the other


`dask.array`
------------

*  Convert numpy/pandas-ish code

        A.dot(B) - B.mean(axis=0)

*  Into Task Graphs

![](images/fail-case.png)


`dask.array`
------------

*  Convert numpy/pandas-ish code

        A.dot(B) - B.mean(axis=0)

*  Then execute those graphs

![](images/fail-case.gif)


### dask.array operations trigger many numpy operations

    $ live demonstration



Use Case: Climate Data
----------------------

We have a pile of NetCDF files

    $ ls
    2014-01-01.nc  2014-03-18.nc  2014-06-02.nc  2014-08-17.nc  2014-11-01.nc
    2014-01-02.nc  2014-03-19.nc  2014-06-03.nc  2014-08-18.nc  2014-11-02.nc
    2014-01-03.nc  2014-03-20.nc  2014-06-04.nc  2014-08-19.nc  2014-11-03.nc
    2014-01-04.nc  2014-03-21.nc  2014-06-05.nc  2014-08-20.nc  2014-11-04.nc
    ...             ...             ...             ...             ...

Four measurements per day, quarter degree resolution, for 2014

    >>> import netCDF4
    >>> t = netCDF4.Dataset('2014-01-01.nc').variables['t2m']
    >>> t.shape
    (4, 721, 1440)


Use Case:  Climate Data
-----------------------

Point to directory of NetCDF datasets

    >>> filenames = sorted(glob('2014-*.nc'))
    >>> temps = [netCDF4.Dataset(fn).variables['t2m'] for fn in filenames]

Wrap each with `dask.array`

    >>> import dask.array as da
    >>> arrays = [da.from_array(t, chunks=(4, 200, 200)) for t in temps]

Manipulate arrays with numpy syntax

    >>> x = da.concatenate(arrays, axis=0)
    >>> x.shape
    (1464, 721, 1440)


Use Case:  Climate Data
-----------------------

Interact with the ecosystem

    >>> from matplotlib import imshow
    >>> imshow(x.mean(axis=0), cmap='bone')

<img src="images/avg.png" width="100%">


Use Case:  Climate Data
-----------------------

Interact with the ecosystem

    >>> from matplotlib import imshow
    >>> imshow(x[1000] - x.mean(axis=0), cmap='RdBu_r')

<img src="images/diff.png" width="100%">


Use Case:  Climate Data
-----------------------

Interact with the ecosystem

    >>> from matplotlib import imshow
    >>> imshow(x[::4].mean(axis=0) - x[2::4].mean(axis=0), cmap='RdBu_r')

<img src="images/day-vs-night.png" width="100%">



Use Case:  TimeSeries Data
--------------------------

    $ live demonstration



### Anatomy of a dask graph


<img src="images/dask-simple.png"
     alt="A simple dask dictionary"
     width="18%"
     align="right">

    # Normal Python             # Dask

    def inc(i):
       return i + 1

    def add(a, b):
       return a + b

    x = 1                       d = {'x': 1,
    y = inc(x)                       'y': (inc, 'x'),
    z = add(y, 10)                   'z': (add, 'y', 10)}

<hr>

    >>> from dask.threaded import get
    >>> get(d, 'z')
    12

*  **Dask graph** is a dictionary of tasks
*  **Task** is a tuple with a callable first element
*  **Arguments** are keys in dictionary ('y') or literal values (10)


<img src="images/dask-simple.png"
     alt="A simple dask dictionary"
     width="18%"
     align="right">

    # Normal Python             # Dask

    def inc(i):
       return i + 1

    def add(a, b):
       return a + b

    x = 1                       d = {'x': 1,
    y = inc(x)                       'y': (inc, 'x'),
    z = add(y, 10)                   'z': (add, 'y', 10)}

<hr>

    >>> from dask.multiprocessing import get
    >>> get(d, 'z')
    12

*  **Dask graph** is a dictionary of tasks
*  **Task** is a tuple with a callable first element
*  **Arguments** are keys in dictionary ('y') or literal values (10)


### Thoughts on Graphs

    d = {'x': 1,
         'y': (inc, 'x'),
         'z': (add, 'y', 10)}

*  Simple representation
*  Easy to engage new developers
*  Generate graphs with straight Python (no DSL)
*  Not user-friendly


### Example - dask.array

    >>> x = da.arange(15, chunks=(5,))
    dask.array<x, shape=(15,), chunks=((5, 5, 5)), dtype=int64>

    >>> x.dask
    {("x", 0): (np.arange,  0,  5),
     ("x", 1): (np.arange,  5, 10),
     ("x", 2): (np.arange, 10, 15)}

    >>> (x + 1000).dask
    {("x", 0): (np.arange,  0,  5),
     ("x", 1): (np.arange,  5, 10),
     ("x", 2): (np.arange, 10, 15),
     ("y", 0): (add, ("x", 0), 1000),
     ("y", 1): (add, ("x", 1), 1000),
     ("y", 2): (add, ("x", 2), 1000)}


### Example - dask.array

    >>> x = da.arange(15, chunks=(5,))
    dask.array<x, shape=(15,), chunks=((5, 5, 5)), dtype=int64>

    >>> x.dask
    {("x", 0): (np.arange,  0,  5),
     ("x", 1): (np.arange,  5, 10),
     ("x", 2): (np.arange, 10, 15)}

    >>> x.sum().dask
    {("x", 0): (np.arange,  0,  5),
     ("x", 1): (np.arange,  5, 10),
     ("x", 2): (np.arange, 10, 15),
     ("y", 0): (np.sum, ("x", 0)),
     ("y", 1): (np.sum, ("x", 1)),
     ("y", 2): (np.sum, ("x", 2)),
     ("y",):   (sum, [("y", 0), ("y", 1), ("y", 2)])}



Compare with Spark
------------------

*  Both expose computation on larger-than-memory data
*  Language Support
    *  Spark: JVM first, Python/R second
    *  Dask: Python only
*  Scale
    *  Spark: Clusters first, single-node second
    *  Dask: Single-node first, clusters maybe some day
*  Graph
    *  Spark: High level RDD graph, then task generation
    *  Dask: Direct to low-level task graphs


### Dask Pleasantness

*  Pure Python / pip installable
*  Productive Workstations
*  Simple graph spec -> easy development
*  Simple graph spec -> easy onboarding
*  Unconstrained graphs -> fun applications

<hr>

### Dask Unpleasantness

*  Distributed scheduling
*  Large graph sizes
*  User defined graphs



User Example: SVD
-----------------

    >>> import dask.array as da
    >>> x = da.ones((5000, 1000), chunks=(1000, 1000))
    >>> u, s, v = da.linalg.svd(x)

<a href="images/dask-svd.png">
  <img src="images/dask-svd.png" alt="Dask SVD graph" width="30%">
</a>

*Work by Mariano Tepper.  "Compressed Nonnegative Matrix Factorization is Fast
and Accurate" [arXiv](http://arxiv.org/abs/1505.04650)*


User Example: SVD
-----------------

    >>> s.dask
    {('x', 0, 0): (np.ones, (1000, 1000)),
     ('x', 1, 0): (np.ones, (1000, 1000)),
     ('x', 2, 0): (np.ones, (1000, 1000)),
     ('x', 3, 0): (np.ones, (1000, 1000)),
     ('x', 4, 0): (np.ones, (1000, 1000)),
     ('tsqr_2_QR_st1', 0, 0): (np.linalg.qr, ('x', 0, 0)),
     ('tsqr_2_QR_st1', 1, 0): (np.linalg.qr, ('x', 1, 0)),
     ('tsqr_2_QR_st1', 2, 0): (np.linalg.qr, ('x', 2, 0)),
     ('tsqr_2_QR_st1', 3, 0): (np.linalg.qr, ('x', 3, 0)),
     ('tsqr_2_QR_st1', 4, 0): (np.linalg.qr, ('x', 4, 0)),
     ('tsqr_2_R', 0, 0): (operator.getitem, ('tsqr_2_QR_st2', 0, 0), 1),
     ('tsqr_2_R_st1', 0, 0): (operator.getitem,('tsqr_2_QR_st1', 0, 0), 1),
     ('tsqr_2_R_st1', 1, 0): (operator.getitem, ('tsqr_2_QR_st1', 1, 0), 1),
     ('tsqr_2_R_st1', 2, 0): (operator.getitem, ('tsqr_2_QR_st1', 2, 0), 1),
     ('tsqr_2_R_st1', 3, 0): (operator.getitem, ('tsqr_2_QR_st1', 3, 0), 1),
     ('tsqr_2_R_st1', 4, 0): (operator.getitem, ('tsqr_2_QR_st1', 4, 0), 1),
     ('tsqr_2_R_st1_stacked', 0, 0): (np.vstack,
                                       [('tsqr_2_R_st1', 0, 0),
                                        ('tsqr_2_R_st1', 1, 0),
                                        ('tsqr_2_R_st1', 2, 0),
                                        ('tsqr_2_R_st1', 3, 0),
                                        ('tsqr_2_R_st1', 4, 0)])),
     ('tsqr_2_QR_st2', 0, 0): (np.linalg.qr, ('tsqr_2_R_st1_stacked', 0, 0)),
     ('tsqr_2_SVD_st2', 0, 0): (np.linalg.svd, ('tsqr_2_R', 0, 0)),
     ('tsqr_2_S', 0): (operator.getitem, ('tsqr_2_SVD_st2', 0, 0), 1)}


Randomized Approximate Parallel Out-of-Core SVD
-----------------------------------------------

    >>> import dask.array as da
    >>> x = da.ones((5000, 1000), chunks=(1000, 1000))
    >>> u, s, v = da.linalg.svd_compressed(x, k=100, n_power_iter=2)

<a href="images/dask-svd-random.png">
<img src="images/dask-svd-random.png"
     alt="Dask graph for random SVD"
     width="10%" >
</a>

N. Halko, P. G. Martinsson, and J. A. Tropp.
*Finding structure with randomness: Probabilistic algorithms for
constructing approximate matrix decompositions.*

*Dask implementation by Mariano Tepper*


Questions?
----------

<hr>

Dask: [dask.pydata.org](http://dask.pydata.org)

*   Useful for:
    *   Multi-core computing
    *   Larger than memory data
    *   Terabyte scale on a workstation
    *   Complex algorithms
    *   NumPy and Pandas interfaces
*   Works by:
    *   High level collections create graphs
    *   Fast scheduler executes graphs in parallel


Questions!
----------

<hr>

Dask: [dask.pydata.org](http://dask.pydata.org)

*   Nuts and bolts of network communication, load balancing, etc..
*   Problems you've seen for which dask is more appropriate
*   Data layout in Tungsten
