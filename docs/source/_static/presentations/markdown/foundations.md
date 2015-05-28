

### `numpy` and `pandas` provide foundational data structures

![](images/jenga.png)


### `numpy` and `pandas` provide composition between different projects

Can use projects together without developer coordination


### But `NumPy` is old

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


### Hardware has changed since 1999

**TODO: Standard image of CPU**

* Multiple cores
   *  4 -- cheap laptop
   *  32 -- workstation
* Distributed memory clusters in big data warehousing
* Fast Solid State Drives (disk as extended memory)


### Python has limitations

* Started in 1991
* Heritage outside of numerics
* Poor support for in-process parallelism
   * The Global Interpreter Lock (GIL) stops two Python threads from
     manipulating Python objects simultaneously


### PyData rests on single-threaded foundations

![](images/jenga.png)

* Incredible domain expertise
* Optimal single-core execution (Scientific heritage)
* But painful to parallelize


### How do we refactor the ecosystem?

![](images/jenga.png)  **TODO: Add Numba**


### Can we parallelize the entire ecosystem in some clever way?


### *no*


### But it might be straightforward


### And we have an effective community
