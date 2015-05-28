
## `dask.array`

*Matthew Rocklin*

Continuum Analytics

[http://dask.pydata.org/](http://dask.pydata.org/)


## I have a big array library

### How could it be more useful to you?


### Related work

*  Parallel BLAS implementations - ScaLAPACK, Plasma, ...
*  Distributed arrays - PETSc/Trillinos, Elemental, HPF
*  Parallel collections - Hadoop/Spark (Dryad, Disco, ...)
*  Task scheduling frameworks - Luigi, swift-lang, ...
*  Python big-numpy projects: Distarray, Spartan, Biggus
*  Custom solutions with MPI, ZMQ, ...

<hr>

### Distinguishing features of `dask.array`

*  Full ndarray support, no serious linear algebra
*  Shared memory parallelism, not distributed
*  Immediately usable - `conda/pip` installable
*  Dask includes other non-array collections



## tl;dr

`dask.array` is...

*  an out-of-core, multi-core, n-dimensional array library
*  that copies the `numpy` interface
*  using blocked algorithms
*  and task scheduling
*  useful for meteorological data


## NumPy interface

`dask.array` supports the following interface from numpy.

*  Arithmetic and scalar mathematics, `+, *, exp, ...`
*  Reductions along axes, `mean(), max(axis=0), ...`
*  Slicing, `x[:100, 500:100:-2]`
*  Fancy indexing, `x[:, [10, 1, 5]]`
*  Some linear algebra, `tensordot, qr, svd`

<hr>

And introduces some novel features

*  Ghosting
*  Integration with HDF5
*  Parallel variants (quantiles, topk, ...)



## Blocked algorithms

Problem -- Given a trillion element array:

*  Find the sum of all elements
*  Find the mean of all elements
*  Find the mean of all positive elements

Solution -- Break array into blocks that fit in-memory.  Use NumPy on each
block.


## Blocked algorithms - Sum

Blocked Sum

    x = h5py.File('myfile.hdf5')['/x']

    sums = []
    for i in range(1000000):
        chunk = x[1000000*i: 1000000*(i+1)]     # Pull out chunk
        sums.append(np.sum(chunk))              # Sum each chunk

    total = sum(sums)                           # Sum intermediate sums


## Blocked algorithms - Mean

Blocked mean of positive elements

    x = h5py.File('myfile.hdf5')['/x']

    sums = []
    counts = []
    for i in range(1000000):
        chunk = x[1000000*i: 1000000*(i+1)]     # Pull out chunk
        chunk = chunk[chunk > 0]                # Filter
        sums.append(np.sum(b))                  # Sum of each chunk
        counts.append(len(b))                   # Count of each chunk

    result = sum(sums) / sum(counts)            # Aggregate results


## Blocked algorithms

Blocked matrix algorithms look like their in-memory equivalents.

Consider matrix multiply:

<img src="images/Matrix_multiplication_diagram.svg.png">


## Blocked algorithms

We didn't need the for loop.

    x = h5py.File('myfile.hdf5')['/x']

    sums = []
    for i in range(1000):
        chunk = x[1000000*i: 1000000*(i+1)]     # Pull out chunk
        sums.append(np.sum(chunk))              # Sum each chunk

    total = sum(sums)                           # Sum intermediate sums

This was parallelizable


## Blocked algorithms

<img src="images/dask_001.png">



## Task scheduling

We execute these graphs with a multi-core scheduler

<img src="images/embarrassing.gif">

And try to keep a small memory footprint


## Task scheduling

Sometimes this fails (but that's ok)

<img src="images/fail-case.gif">



## Meteorological data

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


## Meteorological data

Point to a bunch of NetCDF datasets

    >>> filenames = sorted(glob('2014-*.nc'))
    >>> temps = [netCDF4.Dataset(fn).variables['t2m'] for fn in filenames]

Wrap each with `dask.array`

    >>> import dask.array as da
    >>> arrays = [da.from_array(t, blockshape=(4, 200, 200)) for t in temps]

Manipulate arrays with numpy syntax

    >>> x = da.concatenate(arrays, axis=0)
    >>> x.shape
    (1464, 721, 1440)


## Meteorological data

Interact with the ecosystem

    >>> from matplotlib import imshow
    >>> imshow(x.mean(axis=0), cmap='bone')

<img src="images/avg.png" width="100%">


## Meteorological data

Interact with the ecosystem

    >>> from matplotlib import imshow
    >>> imshow(x[1000] - x.mean(axis=0), cmap='RdBu_r')

<img src="images/diff.png" width="100%">


## Meteorological data

Interact with the ecosystem

    >>> from matplotlib import imshow
    >>> imshow(x[::4].mean(axis=0) - x[2::4].mean(axis=0), cmap='RdBu_r')

<img src="images/day-vs-night.png" width="100%">



## XRay

<img src="images/xray-logo.png" width="30%">

[http://xray.readthedocs.org](http://xray.readthedocs.org)

<hr>

*  Implements the netCDF model
    *  Set of associated ndarrays / variables
    *  Pandas index along each axis
*  Index and reason using named axes with labels
    * NumPy -- `x[40:100].mean(axis=2)`
    * XRay -- `ds.sel(time='2014-04').mean('time')`

<hr>

Written by Stephan Hoyer (@shoyer) at Climate Corp




## Questions?

<img src="images/fail-case.gif">

[http://dask.pydata.org/](http://dask.pydata.org/)


## I have questions

*  What operations do you need?
*  What bandwidths would satisfy you?
*  What hardware do you use?  How flexible are you on this?
