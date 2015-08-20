## `dask.array`

*Matthew Rocklin*

Continuum Analytics

[http://dask.pydata.org/](http://dask.pydata.org/)


## tl;dr

`dask.array` is...

*  an out-of-core, multi-core, n-dimensional array library
*  that copies the `numpy` interface
*  using blocked algorithms
*  and task scheduling


### Related work

*  Parallel BLAS -- ScaLAPACK, Plasma, ...
*  Distributed HPC -- PETSc/Trillinos, Elemental, HPF
*  Parallel Collections -- Hadoop/Spark (Dryad, Disco, ...)
*  Task scheduling -- Luigi, swift-lang, ...
*  Big-NumPy -- Distarray, Spartan, Biggus, Thunder
*  Custom solutions  -- MPI, ZMQ, ...

<hr>

### Distinguishing features of `dask.array`

*  Immediately pip installable
*  Full ndarrays, little serious linear algebra
*  Focus on shared memory / single machine
*  Not just arrays


### NumPy interface

`dask.array` supports the following interface from numpy.

*  Arithmetic -- `+, *, log, exp, ...`
*  Reductions -- `mean(), max(axis=0), ...`
*  Slicing -- `x[:100, 500:100:-2]`
*  Fancy indexing -- `x[:, [10, 1, 5]]`
*  Some linear algebra -- `tensordot, qr, svd`

<hr>

`dask.array` excludes some operations

*  Sort, Eigenvalue solve, Mutation, ...

<hr>

`dask.array` introduces some new operations

*  Parallel algorithms (approximate quantiles, topk, ...)
*  Slightly overlapping arrays
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

    x = h5py.File('myfile.hdf5')['/x']          # Trillion element array on disk

    sums = []
    for i in range(1000000):                    # One million times
        chunk = x[1000000*i: 1000000*(i+1)]     # Pull out chunk
        sums.append(np.sum(chunk))              # Sum chunk

    total = sum(sums)                           # Sum intermediate sums


## Blocked algorithms - Mean

Blocked mean of positive elements

    x = h5py.File('myfile.hdf5')['/x']          # Trillion element array on disk

    sums = []
    counts = []
    for i in range(1000000):                    # One million times
        chunk = x[1000000*i: 1000000*(i+1)]     # Pull out chunk
        chunk = chunk[chunk > 0]                # Filter
        sums.append(np.sum(b))                  # Sum chunk
        counts.append(len(b))                   # Count chunk

    result = sum(sums) / sum(counts)            # Aggregate results


## Blocked algorithms

Consider matrix multiply:

<img src="images/Matrix_multiplication_diagram.svg.png">

Blocked matrix algorithms look like their in-memory equivalents.


## Blocked algorithms

We didn't need the for loop.

    x = h5py.File('myfile.hdf5')['/x']          # Trillion element array on disk

    sums = []
    for i in range(1000000):                    # One million times
        chunk = x[1000000*i: 1000000*(i+1)]     # Pull out chunk
        sums.append(np.sum(chunk))              # Sum chunk

    total = sum(sums)                           # Sum intermediate sums

This was parallelizable


## Blocked algorithms

<img src="images/embarrassing.png">


## Task scheduling

We execute these graphs with a multi-core scheduler

<img src="images/embarrassing.gif">

And try to keep a small memory footprint


## Task scheduling

Sometimes this fails (but that's ok)

<img src="images/fail-case.gif">
