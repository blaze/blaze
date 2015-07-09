## My Job:  Work towards parallel Numeric Python stack


## Python's options for Parallelism

Explicit control.  Fast but hard.

*  Threads/Processes/MPI
*  Concurrent.futures/...
*  Joblib
*  .
*  .
*  .
*  IPython parallel
*  Luigi
*  PySpark
*  Hadoop (mrjob)
*  SQL: Hive, Pig, Impala

Implicit control.  Restrictive/slow but easy.


## Python's options for Parallelism

Explicit control.  Fast but hard.

*  Threads/Processes/MPI
*  Concurrent.futures/...
*  Joblib
*  .
*  .  <-- I need this
*  .
*  IPython parallel
*  Luigi
*  PySpark
*  Hadoop (mrjob)
*  SQL: Hive, Pig, Impala

Implicit control.  Restrictive but easy.


### My Solution: Dynamic task scheduling


## Scale

*  Single four-core laptop (Gigabyte scale)
*  Single thirty-core workstation (Terabyte scale)
*  Distributed thousand-core cluster (Petabyte Scale)


## Scale

*  Single four-core laptop (Gigabyte scale)
*  **Single thirty-core workstation (Terabyte scale)**
*  Distributed thousand-core cluster (Petabyte Scale)


## Outline

*  Dask.array - parallel array library using dask
*  Dask - internals
*  Dask.dataframe/other - think about if this is useful to you
