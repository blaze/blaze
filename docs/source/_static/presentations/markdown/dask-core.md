## Dask

Dead simple task scheduling


## We've seen `dask.array`

*  Turns Numpy-ish code

        (2*x + 1) ** 3

*  Into Graphs

![](images/dask_001.png)


## We've seen `dask.array`

*  Turns Numpy-ish code

        (2*x + 1) ** 3

*  Into Graphs - and then executes those graphs

![](images/embarrassing.gif)


## Works for more than just arrays

    import dask.bag as db

    b = db.from_filenames('data/2014-*.json.gz').map(json.loads)

    b.groupby('username')

![](images/dask-bag-shuffle.png)


*  Collections build graphs
*  Schedulers execute graphs

![](images/collections-schedulers.png)

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

    d = {'x': 1,
         'y': (inc, 'x'),
         'z': (add, 'y', 10)}

* Schedulers manage execution


### Dask's schedulers enable sane parallelism

### ... even if your workflow isn't just arrays


### Example

    def load(filename):
        ...
    def clean(data):
        ...
    def analyze(sequence_of_data):
        ...
    def store(result):
        ...

    dsk = {'load-1': (load, 'myfile.a.data'),
           'load-2': (load, 'myfile.b.data'),
           'load-3': (load, 'myfile.c.data'),
           'preprocess-1': (clean, 'load-1'),
           'preprocess-2': (clean, 'load-2'),
           'preprocess-3': (clean, 'load-3'),
           'analyze': (analyze, ['preprocess-%d' % i for i in [1, 2, 3]]),
           'store': (store, 'analyze')}

*  Not for end users
*  Maybe for library developers
*  Regardless, the community should search for a solution
