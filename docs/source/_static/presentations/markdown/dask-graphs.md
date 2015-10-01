## Common Communication Patterns with Dask Arrays

Dask arrays/frames provide translation from NumPy/Pandas syntax to visual
blocked algorithms.


### Make a dask array of ones

    >>> import dask.array as da
    >>> x = da.ones(15, chunks=(5,))

### And visualize the resulting dask graph

    >>> x.visualize('dask.pdf')

![](images/dask.ones.png)

We're going to do this for increasingly complex expressions which create
increasingly complex blocked algorithm task graphs.


### Elementwise operations

    >>> x + 1

![](images/dask.ones-plus-one.png)


### Elementwise operations

    >>> (x + 1) * 2

![](images/dask.ones2.png)


### Reductions

    >>> (x + 1).sum()

![](images/dask.ones-sum.png)


### Slicing

    >>> (x + 1)[3:9].sum()

<img src="images/dask.ones-slice-sum.png"
     width="40%">


### Ghosting (shared boundaries)

    >>> x = da.ones(100, chunks=(10,))
    >>> g = da.ghost.ghost(x, depth={0: 2}, boundary={0: np.nan})

![](images/dask.ghost.png)



### Two Dimensional Algorithms

    >>> x = da.ones((15, 15), chunks=(5, 5))


### Partial Reductions

    >>> x.mean(axis=0)

![](images/dask.2d-mean.png)


### Transpose

    >>> x + x.T

![](images/dask.2d-transpose.png)


### Matrix Multiply (index contraction)

    >>> x.dot(x.T)

![](images/dask.2d-dot.png)


### Compound ad naseum

    >>> x.dot(x.T + 1) - x.mean(axis=1)

![](images/dask.2d-compound.png)

We can compound these operations forever.  Constructing larger and larger
graphs before we hand off the work to a scheduler to execute.
