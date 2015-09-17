dask: Parallel PyData
---------------------

*Matthew Rocklin*

Continuum Analytics


Goals
-----

* PyData Stack on Medium Data (10GB - 10TB)
* Multi-core
* Larger-than-memory
* Avoid Setup (single-node)
* Complex Algorithms


Solution
--------

*   High Level: User libraries that mimic PyData libraries

    dask.array, dask.dataframe, dask.imperative

*   Task Dependency Graphs

*   Low Level: Scheduler low-latency, parallel, memory aware


Blocked Algorithms with dask.array
----------------------------------

*  Mimic the Numpy Interface
*  Operate on larger-than-memory data
*  Leverage all available cores
*  Use NumPy under the hood

<hr>

### dask.array operations trigger many numpy operations


Blocked Algorithms with dask.array
----------------------------------

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


How does this work?
-------------------

*  Convert numpy/pandas-ish code

        A.dot(B) - B.mean(axis=0)

*  Into Task Graphs

![](images/fail-case.png)


How does this work?
-------------------

*  Convert numpy/pandas-ish code

        A.dot(B) - B.mean(axis=0)

*  Then execute those graphs

![](images/fail-case.gif)


Ease
----

*  Works well on a laptop or workstation
*  Avoids memory blowup where possible
*  Leverage many cores
*  No JVM boundary
*  Easy to extend



## User Example: SVD

    >>> import dask.array as da
    >>> x = da.ones((5000, 1000), chunks=(1000, 1000))
    >>> u, s, v = da.linalg.svd(x)

<a href="images/dask-svd.png">
  <img src="images/dask-svd.png" alt="Dask SVD graph" width="30%">
</a>

*Work by Mariano Tepper.  "Compressed Nonnegative Matrix Factorization is Fast
and Accurate" [arXiv](http://arxiv.org/abs/1505.04650)*


## SVD - Parallel Profile

<iframe src="svd.profile.html"
        marginwidth="0"
        marginheight="0" scrolling="no" width="800"
        height="300"></iframe>


## Randomized Approximate Parallel Out-of-Core SVD

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
    *   Process a terabyte on your laptop
    *   Complex algorithms (not a database)
    *   NumPy and Pandas interfaces
*   Works by:
    *   High level collections create graphs
    *   Fast scheduler executes graphs in parallel
