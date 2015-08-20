### Python's options for Parallelism

Explicit control -- Fast but hard

*  Threads/Processes/MPI/ZeroMQ
*  Concurrent.futures/Joblib/...
*  .
*  .
*  .
*  IPython parallel
*  Luigi
*  PySpark
*  Hadoop (mrjob)
*  SQL: Hive, Pig, Impala

Implicit control -- Restrictive but easy


### Python's options for Parallelism

Explicit control -- Fast but hard

*  Threads/Processes/MPI/ZeroMQ
*  Concurrent.futures/Joblib/...
*  .
*  .  <-- I need this
*  .
*  IPython parallel
*  Luigi
*  PySpark
*  Hadoop (mrjob)
*  SQL: Hive, Pig, Impala

Implicit control -- Restrictive but easy


### Scale

*  Single four-core laptop (Gigabyte scale)
*  Single thirty-core workstation (Terabyte scale)
*  Distributed thousand-core cluster (Petabyte Scale)


### Scale

*  **Single four-core laptop (Gigabyte scale)**
*  **Single thirty-core workstation (Terabyte scale)**
*  Distributed thousand-core cluster (Petabyte Scale)


### Upcoming Outline

*  Dask.array - parallel array library using dask
*  Dask internals - dynamic task scheduling library
*  Beyond arrays - How can we extend task parallelism
