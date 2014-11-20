## Expression Chunking


### Suppose we have a large array of integers

A trillion numbers

    x = array([5, 3, 1, ... <one trillion numbers>, ... 12, 5, 10])

How do we compute the largest?

    x.max()


### Max by Chunking

    size = 1000000
    chunk = x[size * i: size * (i + 1)]

Max of each chunk

    aggregate[i] = chunk.max()

Max of aggregated results

    aggregate.max()


### Sum by Chunking

    size = 1000000
    chunk = x[size * i: size * (i + 1)]

Sum of each chunk

    aggregate[i] = chunk.sum()

Sum of aggregated results

    aggregate.sum()


### Count by Chunking

    size = 1000000
    chunk = x[size * i: size * (i + 1)]

Count each chunk

    aggregate[i] = chunk.count()

Sum aggregated results

    aggregate.sum()


### Mean by Chunking

    size = 1000000
    chunk = x[size * i: size * (i + 1)]

Sum and count of each chunk

    aggregate.total[i] = chunk.sum()
    aggregate.n[i] = chunk.count()

Sum the total and count then divide

    aggregate.total.sum() / aggregate.n.sum()


### Number of occurrences by Chunking

    size = 1000000
    chunk = x[size * i: size * (i + 1)]

Split-apply-combine on each chunk

    by(x, freq=x.count())

Split-apply-combine on concatenation of results

    by(aggregate, freq=aggregate.freq.sum())
