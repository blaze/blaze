## Final Thoughts

*  Python and Parallelism
    *  Most data is small
    *  Storage, representation, streaming, sampling offer bigger gains
    *  That being said, please [release the GIL](https://github.com/scikit-image/scikit-image/pull/1519)

*  Dask: Dynamic task scheduling yields sane parallelism
    *  Simple library to enable parallelism
    *  Dask.array/dataframe demonstrate ability
    *  Rarely optimal performance (Theano is far smarter)
    *  Scheduling necessary for composed algorithms

*  Questions:
    *  Appropriate class of problems in ML?
    *  What is the right API for algorithm builders?


## Questions?

[http://dask.pydata.org](http://dask.pydata.org)

<img src="images/jenga.png" width="60%">

<img src="images/fail-case.gif" width="60%">
