## Expression Chunking


## ... designing parallel algorithms


## ... out-of-core, parallel, numpy/pandas


### Suppose we have a large array of integers

A trillion numbers

    x = np.array([5, 3, 1, ... <one trillion numbers>, ... 12, 5, 10])

How do we compute the largest?

    x.max()

<hr>

### Define the problem in Blaze

    >>> from blaze import symbol
    >>> x = symbol('x', '1000000000 * int')
    >>> x.max()


### Max by Chunking

    size = 1000000
    chunk = x[size * i: size * (i + 1)]

Max of each chunk

    aggregate[i] = chunk.max()

Max of aggregated results

    aggregate.max()

<hr>

    >>> from blaze.expr.split import split
    >>> split(x, x.max())
    ((chunk,     max(chunk)),
     (aggregate, max(aggregate)))


### Sum by Chunking

    size = 1000000
    chunk = x[size * i: size * (i + 1)]

Sum of each chunk

    aggregate[i] = chunk.sum()

Sum of aggregated results

    aggregate.sum()

<hr>

    >>> from blaze.expr.split import split
    >>> split(x, x.sum())
    ((chunk,     sum(chunk)),
     (aggregate, sum(aggregate)))


### Count by Chunking

    size = 1000000
    chunk = x[size * i: size * (i + 1)]

Count each chunk

    aggregate[i] = chunk.count()

Sum aggregated results

    aggregate.sum()

<hr>

    >>> from blaze.expr.split import split
    >>> split(x, x.count())
    ((chunk,     count(chunk)),
     (aggregate, sum(aggregate)))


### Mean by Chunking

    size = 1000000
    chunk = x[size * i: size * (i + 1)]

Sum and count of each chunk

    aggregate.total[i] = chunk.sum()
    aggregate.n[i] = chunk.count()

Sum the total and count then divide

    aggregate.total.sum() / aggregate.n.sum()

<hr>

    >>> from blaze.expr.split import split
    >>> split(x, x.mean())
    ((chunk,     summary(count=count(chunk), total=sum(chunk))),
     (aggregate, sum(aggregate.total)) / sum(aggregate.count))


### Number of occurrences by Chunking

    size = 1000000
    chunk = x[size * i: size * (i + 1)]

Split-apply-combine on each chunk

    by(x, freq=x.count())

Split-apply-combine on concatenation of results

    by(aggregate, freq=aggregate.freq.sum())

<hr>

    >>> from blaze.expr.split import split
    >>> split(x, by(x, freq=x.count())
    ((chunk,     by(chunk, freq=count(chunk))),
     (aggregate, by(aggregate.chunk, freq=sum(aggregate.freq))))


### N-Dimensional reductions

Data: a  10000 by 10000 by 10000 array of (x,y) coordinates

    >>> points = symbol('points', '10000 * 10000 * 10000 * {x: int, y: int}')

Chunk: a cube of a billion elements

    >>> chunk = symbol('chunk', '1000 * 1000 * 1000 * {x: int, y: int}')

Expr: The variance of their addition

    >>> expr = (points.x + points.y).var(axis=0)
    >>> split(points, expr, chunk=chunk)
    ((chunk,
      summary(n  = count( chunk.x + chunk.y ),
              x  =   sum( chunk.x + chunk.y ),
              x2 =   sum((chunk.x + chunk.y) ** 2))),
     (aggregate,
        (sum(aggregate.x2) / (sum(aggregate.n)))
     - ((sum(aggregate.x)  / (sum(aggregate.n))) ** 2)))

Known shapes:

    >>> aggregate.dshape
    dshape("10 * 10 * 10 * {n: int32, x: float64, x2: float64}")


### Recap

Blaze expressions let us design powerful algorihtms abstractly.  Development is
fast and generally applicable.

<hr>

### Limitations

* No sorting, joining, etc..

* Only single-dataset operations (notably missing dot products)

* Only a third of a solution.
    *   Expression splitting - *what* do we want to compute?
    *   Task scheduling - *where* do we compute each piece?
    *   In-memory execution - *how* do we actually execute this?
