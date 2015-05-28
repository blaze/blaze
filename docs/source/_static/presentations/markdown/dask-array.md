
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


## NumPy interface

`dask.array` supports the following interface from numpy.

*  Arithmetic and scalar mathematics, `+, *, exp, ...`
*  Reductions along axes, `mean(), max(axis=0), ...`
*  Slicing, `x[:100, 500:100:-2]`
*  Fancy indexing, `x[:, [10, 1, 5]]`
*  Some linear algebra, `tensordot, qr, svd`

<hr>

And introduces some novel features

*  Overlapping boundaries
*  Parallel variant algorithms (quantiles, topk, ...)
*  Integration with HDF5



## Blocked algorithms

*  Problem -- Given a trillion element array:
    *  Find the sum of all elements
    *  Find the mean of all elements
    *  Find the mean of all positive elements
*   Solution -- Break array into blocks that fit in-memory.

    Use NumPy on each block.


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


## Notable Users

*   [xray](http://xray.readthedocs.org)
    by [Stephan Hoyer](http://http://stephanhoyer.com/)
    at Climate Corp
*   [Scikit-image](http://scikit-image.org/) for parallelizing some filters
*   [Mariano Tepper](http://www.marianotepper.com.ar/) (Maths postdoc at Duke)
    builds `dask.array.linalg`
*   I use it daily for internal projects
