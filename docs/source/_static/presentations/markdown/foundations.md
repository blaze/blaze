### PyData builds off of NumPy and Pandas


### NumPy and Pandas provide foundational data structures

<img src="images/jenga.png" width="100%">


### Data structures enable composition

### ... cross-project interactions without coordination


### But NumPy is old

```
mrocklin@notebook:~/scipy$ git log | tail

Author: Travis Oliphant <oliphant@enthought.com>
Date:   Fri Feb 2 05:08:11 2001 +0000

    shouldn't work

commit 02de46a5464f182d3d64be5a7ee1087ae8be8646
Author: Eric Jones <eric@enthought.com>
Date:   Thu Feb 1 08:32:30 2001 +0000

    Initial revision
```


### NumPy and Pandas have limitations

*  Single Threaded (mostly)
*  In-memory data (mostly)
*  Poor support for variable length strings
*  Poor support for missing data
*  ...

### These limitations affect the PyData ecosystem


### Hardware has changed since 2001

![](images/multicore-cpu.png)

* Multiple cores
   *  4 cores -- cheap laptop
   *  32 cores -- workstation
* Distributed memory clusters in big data warehousing
* Fast Solid State Drives (disk is now extended memory)


### Hardware has changed since 2001

![](images/xeon-phi.jpg)

* Multiple cores
   *  4 cores -- cheap laptop
   *  32 cores -- workstation
* Distributed memory clusters in big data warehousing
* Fast Solid State Drives (disk is now extended memory)


### Problems have changed since 2001

*  Larger datasets
*  Messier data
*  More text data


### Python has limitations

* Started in 1991
* Heritage outside of numerics
* Poor support for in-process parallelism

<hr>

### Global Interpreter Lock

*  The Global Interpreter Lock (GIL) stops two Python threads from
   manipulating Python objects simultaneously
*  Solutions:
    * Compute in separate processes (hard to share data)
    * Release the GIL and use C/Fortran code


### PyData rests on single-threaded foundations

<img src="images/jenga.png" width="100%">

* Incredible domain expertise
* Optimal single-core execution (Scientific heritage)
* But painful to parallelize


### Can we parallelize the ecosystem without touching downstream projects?


### *probably not*


### But this work might be straightforward


### And we have an effective community
