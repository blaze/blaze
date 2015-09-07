### Scikit Image Case Study

*  Setup
    *  Scikit image has sophisticated single-threaded algorithms
    *  Dask.array parallelizes map on slighty overlapping blocks

<img src="images/ghosted-neighbors.png">

*  Timeline
    *  [Blake Griffith](http://github.com/cowlicks/)
       creates
       [parallel apply function](https://github.com/scikit-image/scikit-image/pull/1493)
       in scikit-image (1 week part time)
    *  People try it out; it's not much faster
    *  [Johannes Schönberger](http://www.cs.unc.edu/~jsch/) [releases the GIL](https://github.com/scikit-image/scikit-image/pull/1519/files) (few days)
    *  Scikit image + dask.array sees
        [2x-3x speedups](https://github.com/blaze/dask/blob/master/notebooks/parallelize_image_filtering_workload.ipynb)
        over Scikit image alone  (experiments by [@arve0](http://arve0.github.io/))


### Momentum

*   Jeff Reback has [a nogil Pandas branch](https://github.com/pydata/pandas/pull/10199)

    This morning: *I updated this. works for all groupbys now.*

*   [Bottleneck issue](https://github.com/kwgoodman/bottleneck)


### Final thoughts

[http://dask.pydata.org](http://dask.pydata.org)

*  Most data is small (*you should ignore this talk*)
*  PyData has room to grow in parallelism (GIL is not an issue)
*  Dask.array -- a multi-core on-disk numpy clone
*  Dask.core -- an option for parallelism

<img src="images/fail-case.gif" width="70%">


### Finally: Parallelism is rarely important

*  Most data is small
*  For moderate data, think about storage and representation
*  Pandas categoricals are possibly the biggest improvement to PyData performance in
   the last year


### Ignore everything I just said

*  Most data is small
*  For moderate data, think about storage and representation
*  Pandas categoricals are possibly the biggest improvement to PyData performance in
   the last year


### Questions?

[http://dask.pydata.org](http://dask.pydata.org)

![](images/jenga.png)

![](images/collections-schedulers.png)
