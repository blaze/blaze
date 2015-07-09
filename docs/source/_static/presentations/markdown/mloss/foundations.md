### Numeric Python Stack

<hr>

### C/Fortran, NumPy, Pandas


### NumPy and Pandas provide foundational data structures

<img src="images/jenga.png" width="100%">


### Shared data structures enable interactions without coordination


### Enables a vibrant ecosystem

### but exposes us to risk of obsolescence


### Python, NumPy and Pandas are old(ish)

*  Python: 1991
*  NumPy: 1995
*  Pandas: 2008


### NumPy and Pandas have limitations

*  Single Threaded (mostly)
*  In-memory data (mostly)
*  Poor support for variable length strings
*  Poor support for missing data
*  Poor support for nested/semi-structured data


### The Numeric Python ecosystem inherits these limitations


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


## Why do we still use Python?

*  Easy to setup and use by domain scientists
*  C/Fortran heritage
*  Hundreds of PhD theses in software stack
*  Strong academic and industry communities


### PyData rests on single-threaded foundations

<img src="images/jenga.png" width="100%">

*  Incredible domain expertise
*  Optimal single-core execution (scientific heritage)
*  But painful to parallelize


### Q: Can we parallelize the ecosystem without touching downstream projects?


### *probably not*


### But this work might be straightforward


### And we have an effective community
