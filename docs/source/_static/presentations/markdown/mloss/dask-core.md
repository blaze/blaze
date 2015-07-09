## `dask.core`

Dead simple task scheduling

[dask.pydata.org](http://dask.pydata.org/en/latest/)


## We've seen `dask.array`

*  Turns Numpy-ish code

        (2*x + 1) ** 3

*  Into Graphs

![](images/embarrassing.png)


## We've seen `dask.array`

*  Turns Numpy-ish code

        (2*x + 1) ** 3

*  Then executes those graphs

![](images/embarrassing.gif)


### Q: What constitutes a dask graph?


<img src="images/dask-simple.png"
     alt="A simple dask dictionary"
     width="18%"
     align="right">

    # Normal Python             # Dask

    def inc(i):
       return i + 1

    def add(a, b):
       return a + b

    x = 1                       d = {'x': 1,
    y = inc(x)                       'y': (inc, 'x'),
    z = add(y, 10)                   'z': (add, 'y', 10)}

<hr>

    >>> from dask.threaded import get
    >>> get(d, 'z')
    12

*  **Dask graph** is a dictionary of tasks
*  **Task** is a tuple with a callable first element
*  **Arguments** are keys in dictionary ('y') or literal values (10)


<img src="images/dask-simple.png"
     alt="A simple dask dictionary"
     width="18%"
     align="right">

    # Normal Python             # Dask

    def inc(i):
       return i + 1

    def add(a, b):
       return a + b

    x = 1                       d = {'x': 1,
    y = inc(x)                       'y': (inc, 'x'),
    z = add(y, 10)                   'z': (add, 'y', 10)}

<hr>

    >>> from dask.multiprocessing import get
    >>> get(d, 'z')
    12

*  **Dask graph** is a dictionary of tasks
*  **Task** is a tuple with a callable first element
*  **Arguments** are keys in dictionary ('y') or literal values (10)


### Thoughts on Graphs

    d = {'x': 1,
         'y': (inc, 'x'),
         'z': (add, 'y', 10)}

*  Simple representation
*  Use normal Python code to generate graphs (no DSL)
*  Not user-friendly


### Example - dask.array

    >>> x = da.arange(15, chunks=(5,))
    dask.array<x, shape=(15,), chunks=((5, 5, 5)), dtype=None>

    >>> x.dask
    {("x", 0): (np.arange,  0,  5),
     ("x", 1): (np.arange,  5, 10),
     ("x", 2): (np.arange, 10, 15)}

    >>> x.sum().dask
    {("x", 0): (np.arange,  0,  5),
     ("x", 1): (np.arange,  5, 10),
     ("x", 2): (np.arange, 10, 15),
     ("s", 0): (np.sum, ("x", 0)),
     ("s", 1): (np.sum, ("x", 1)),
     ("s", 2): (np.sum, ("x", 2)),
     ("s",):   (sum, [("s", 0), ("s", 1), ("s", 2)])}


### Dask.array is a convenient way to make dictionaries

<hr>

### Dask is a convenient way to make libraries like dask.array


## Dask works for more than just arrays

* `dask.array` = `numpy` + `threading`
* `dask.bag` = `list` + `multiprocessing`
* `dask.dataframe` = `pandas` + `threading`
* ...


*  Collections build graphs
*  Schedulers execute graphs

<img src="images/collections-schedulers.png"
     width="100%">

*  Neither side needs the other


### Question: Is there a similar class of problems in ML?

<hr>

### Question: How should we write them down as code?


### Up Next:  Example projects using dask to parallelize other workflows
