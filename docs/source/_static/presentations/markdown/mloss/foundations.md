### Numeric Python Stack

<hr>

### C/Fortran, NumPy, SciPy, Pandas


### NumPy and Pandas provide foundational data structures

<img src="images/jenga.png" width="100%">


### Shared data structures enable interactions


### Shared data structures enable a vibrant ecosystem

### but expose us to risk of rigidity


### Python, NumPy and Pandas are old(ish)

*  Python: 1991
*  Numeric+NumPy: 1995
*  Pandas: 2008


### Python, NumPy and Pandas are old(ish)

*  Python: 1991 (Super Nintendo)
*  Numeric+NumPy: 1995 (Playstation)
*  Pandas: 2008 (iPhone)


### NumPy and Pandas have limitations

*  Single Threaded (mostly)
*  In-memory data (mostly)
*  Poor support for variable length strings
*  Poor support for missing data
*  Poor support for nested/semi-structured data
*  Code bases are now hard to change


### The Numeric Python ecosystem inherits these limitations


### Python has limitations

* Started in 1991
* Heritage outside of numerics
* Poor support for in-process parallelism

<hr>

### Global Interpreter Lock

*  The Global Interpreter Lock (GIL) stops two Python threads from
   manipulating Python objects simultaneously
*  Solutions:
    * Use separate processes (hard to share data)
    * Use C/Fortran code and release the GIL
        * Pandas releases the GIL on groupby operations
        * Scikit-image [released the GIL](https://github.com/scikit-image/scikit-image/pull/1519) in a weekend sprint


### Hardware has changed since 1991

![](images/multicore-cpu.png)

* Multiple cores
   *  4 cores -- cheap laptop
   *  32 cores -- workstation
   * 1000 cores -- distributed memory big data warehouses
   * 1e6 core -- HPC Supercomputer
* Fast Disk -- SSDs extend memory


### Hardware has changed since 1991

![](images/xeon-phi.jpg)


## Why do we still use Python?

*  Ubiquitous
*  Easy to setup and use
*  C/Fortran heritage
*  NumPy and Pandas are still pretty awesome
*  Domain expertise in the software stack (scikits)
*  Strong academic and industry relationship
*  Other communities (web, dev-ops, etc..)


### PyData rests on single-threaded foundations

<img src="images/jenga.png" width="100%">

*  Incredible domain expertise
*  Optimal single-core execution (scientific heritage)
*  But painful to parallelize


### Q: Can we parallelize the ecosystem without touching downstream projects?


### *probably not*


### But this work might be straightforward


### And we have an effective community
