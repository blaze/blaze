## Dask Arrays

*or*

## PyData's Relationship with Parallelism

*Matthew Rocklin*

Continuum Analytics


## Outline

* .
* Dask.array
    *  Multicore parallelism with blocked algorithms
    *  Out-of-core execution with task scheduling
* .
* .


## Outline

* PyData's uneasy relationship with parallelism
* Dask.array
    *  Multicore parallelism with blocked algorithms
    *  Out-of-core execution with task scheduling
* Dask.core
    *  Extend parallelism to other contexts
* PyData and the GIL


## Parallelism and Data

*  Gigabyte - Fits in memory, need one core  (laptop)
*  Terabyte - Fits on disk, need ten cores  (workstation)
*  Petabyte - Fits on many disks, need 1000 cores (cluster)
