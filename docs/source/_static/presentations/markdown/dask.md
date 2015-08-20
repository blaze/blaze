### Dask - Task scheduling and Large Arrays

![](images/fail-case.gif)

    expr = x.T.dot(y) - y.mean(axis=0)


### NumPy

* NumPy powers the scientific software stack
    * Pandas
    * SciPy
    * Matplotlib
    * Scikit learn, image, ...

```python
>>> import numpy as np
>>> x = np.load(...)
>>> y = ...

>>> x.T.dot(y) - y.mean(axis=0)  # Complex, expressive, fast
```

* But NumPy is (mostly) restricted to memory and a single core
    * Along with the rest of the stack


### ... this is usually fine

*99% of problems fit in memory*


### `dask.array`

*  Implement blocked array algorithms
*  is a drop in replacement for a subset of NumPy
*  Keeps a small memory footprint
*  Uses all of your cores

```python
>>> import h5py
>>> d = h5py.File('myfile.hdf5')['/my/huge/array']  # a giant on-disk array
>>> d.shape
(1000000, 1000000)

>>> import dask.array as da
>>> x = da.from_array(d, chunks=(1000, 1000))   # cut up array into blocks

>>> y = x.T.dot(x).mean(axis=0)                     # do numpy math
>>> plot(y[::100])                                  # use result as normal
```


### But first, `dask`

<img src="http://dask.readthedocs.org/en/latest/_images/dask-simple.png"
     align="right">

* Consider the following program:

```python
def inc(i):
    return i + 1

def add(a, b):
    return a + b

x = 1
y = inc(x)
z = add(y, 10)
```

* Encode as a dictionary:

```python
d = {'x': 1,
     'y': (inc, 'x'),
     'z': (add, 'y', 10)}
```


### We choose how and when to execute this code.

* Dask graph

```python
d = {'x': 1,
     'y': (inc, 'x'),
     'z': (add, 'y', 10)}
```

* Simple scheduler / execution

```python
>>> dask.core.get(d, 'x')
1
>>> dask.core.get(d, 'z')
12
```

* Use different schedulers for different hardware


### Dask arrays create graphs from numpy-like code

*live demo*


### Execute results with asynchronous scheduler

![](images/fail-case.gif)


### Example: Stack of Meteorological Data

    $ ls
    2014-01-01.nc3  2014-03-18.nc3  2014-06-02.nc3  2014-08-17.nc3  2014-11-01.nc3
    2014-01-02.nc3  2014-03-19.nc3  2014-06-03.nc3  2014-08-18.nc3  2014-11-02.nc3
    2014-01-03.nc3  2014-03-20.nc3  2014-06-04.nc3  2014-08-19.nc3  2014-11-03.nc3
    2014-01-04.nc3  2014-03-21.nc3  2014-06-05.nc3  2014-08-20.nc3  2014-11-04.nc3
    ...             ...             ...             ...             ...

```python
>>> import netCDF4
>>> t = netCDF4.Dataset('2014-01-01.nc3').variables['t2m']
>>> t.shape
(4, 721, 1440)
```


### Collect all temperature data

```python
>>> from glob import glob
>>> filenames = sorted(glob('2014-*.nc3'))
>>> temps = [netCDF4.Dataset(fn).variables['t2m'] for fn in filenames]
```

### Concatenate with `dask.array`

```python
>>> import dask.array as da
>>> arrays = [da.from_array(t, chunks=(4, 200, 200)) for t in temps]
>>> x = da.concatenate(arrays, axis=0)

>>> x.shape
(1464, 721, 1440)
```


### Plot

```python
>>> imshow(x.mean(axis=0), cmap='bone')
>>> imshow(x[1000] - x.mean(axis=0), cmap='RdBu_r')
```

<img src="images/avg.png" align="left" width="45%">
<img src="images/diff.png" align="right" width="45%">


### Plot

```python
>>> imshow(x[::4].mean(axis=0) - x[2::4].mean(axis=0), cmap='RdBu_r')
```

<img src="images/day-vs-night.png" align="right" width="90%">


### Questions?

*  Source:  [http://github.com/blaze/dask/](http://github.com/blaze/dask/)
*  Docs:  [http://dask.readthedocs.org](http://dask.readthedocs.org)

![](images/fail-case.gif)

    expr = x.T.dot(y) - y.mean(axis=0)
