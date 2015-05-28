## Dask

Dead simple task scheduling


## We've seen `dask.array`

*  Turns Numpy-ish code

        (2*x + 1) ** 3

*  Into Graphs

![](images/dask_001.png)


## We've seen `dask.array`

*   .

        .

*  Then executes those graphs

![](images/embarrassing.gif)


## Works for more than just arrays

    import dask.bag as db

    b = db.from_filenames("data/2014-*.json.gz").map(json.loads)

    b.groupby("username")

![](images/dask-bag-shuffle.png)


*  Collections build graphs
*  Schedulers execute graphs

<img src="images/collections-schedulers.png"
     width="100%">

*  Neither side needs the other


<img src="images/dask-simple.png"
     alt="A simple dask dictionary"
     align="right">

### Normal Python

    def inc(i):
       return i + 1

    def add(a, b):
       return a + b

    x = 1
    y = inc(x)
    z = add(y, 10)

* CPython manages execution

<hr>

### Dask graph

    d = {"x": 1,
         "y": (inc, "x"),
         "z": (add, "y", 10)}

* Schedulers manage execution


### Example - dask.array

    >>> import dask.array as da

    >>> x = da.arange(15, chunks=(5,))
    dask.array<x, shape=(15,), chunks=((5, 5, 5)), dtype=None>

    >>> x.dask
    {("x", 0): (np.arange,  0,  5),
     ("x", 1): (np.arange,  5, 10),
     ("x", 2): (np.arange, 10, 15)}

    >>> (x + 100).dask
    {("x", 0): (np.arange,  0,  5),
     ("x", 1): (np.arange,  5, 10),
     ("x", 2): (np.arange, 10, 15),
     ("y", 0): (add, ("x", 0), 100),
     ("y", 1): (add, ("x", 1), 100),
     ("y", 2): (add, ("x", 2), 100)}


### Example - dask.array

    >>> import dask.array as da

    >>> x = da.arange(15, chunks=(5,))
    dask.array<x, shape=(15,), chunks=((5, 5, 5)), dtype=None>

    >>> x.dask
    {("x", 0): (np.arange,  0,  5),
     ("x", 1): (np.arange,  5, 10),
     ("x", 2): (np.arange, 10, 15)}

    >>> x.sum()
    {("x", 0): (np.arange,  0,  5),
     ("x", 1): (np.arange,  5, 10),
     ("x", 2): (np.arange, 10, 15),
     ("y", 0): (np.sum, ("x", 0)),
     ("y", 1): (np.sum, ("x", 1)),
     ("y", 2): (np.sum, ("x", 2)),
     ("z",): (np.sum, [("y", 0), ("y", 1), ("y", 2)])}


### Example - custom graph

    def load(filename):
        ...
    def clean(data):
        ...
    def analyze(sequence_of_data):
        ...
    def store(result):
        ...

    dsk = {"load-1": (load, "myfile.a.data"),
           "load-2": (load, "myfile.b.data"),
           "load-3": (load, "myfile.c.data"),
           "preprocess-1": (clean, "load-1"),
           "preprocess-2": (clean, "load-2"),
           "preprocess-3": (clean, "load-3"),
           "analyze": (analyze, ["preprocess-%d" % i for i in [1, 2, 3]]),
           "store": (store, "analyze")}


### Example - custom graph

    dsk = {"load-1": (load, "myfile.a.data"),
           "load-2": (load, "myfile.b.data"),
           "load-3": (load, "myfile.c.data"),
           "preprocess-1": (clean, "load-1"),
           "preprocess-2": (clean, "load-2"),
           "preprocess-3": (clean, "load-3"),
           "analyze": (analyze, ["preprocess-%d" % i for i in [1, 2, 3]]),
           "store": (store, "analyze")}

    from dask.multiprocessing import get
    result = get(dsk, ["store"])


### Dask's schedulers enable sane parallelism

### ... even if your workflow isn't arrays

*  Dead simple way to describe data dependencies
*  Relies on battle-tested schedulers
*  Raw dicts probably not for end users
*  But maybe for library developers
*  Regardless, the community should search for a parallelism abstraction
