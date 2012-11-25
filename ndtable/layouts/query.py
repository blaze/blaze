import numpy as np

def getitem(cc, indexer, data):

    # Fancy Indexing [...]
    if isinstance(indexer, tuple):

        # Fancy Indexing [(a,b)]
        if len(indexer) == 2:
            a = indexer[0]
            b = indexer[1]

            max1 = data.bounds[0]
            max2 = data.bounds[1]

            ix = range(a.start or 0, a.stop or max1, a.step or 1)
            iy = range(a.start or 0, b.stop or max2, b.step or 1)

            # TODO: use source.empty() to generalize
            res = np.empty((len(ix), len(iy)))

            for a, i in enumerate(ix):
                for b, j in enumerate(iy):
                    res[a,b] = data[cc(i,j)]
            return res

        # Fancy Indexing [a]
        elif len(indexer) == 1:
            return data[cc(indexer)]

        else:
            raise NotImplementedError

def setitem(cc, indexer, data, value):

    # [(a,b)]
    if hasattr(indexer, '__iter__'):
        if isinstance(indexer[0], slice):
            if len(indexer) == 2:
                idx0 = indexer[0]
                idx1 = indexer[1]

                max1 = data.bounds1
                max2 = data.bounds1

                ix = range(idx0.start or 0, idx0.stop or max1, idx0.step or 1)
                iy = range(idx1.start or 0, idx1.stop or max2, idx1.step or 1)

                if hasattr(value, '__iter__'):
                    viter = iter(value)

                    for a, i in enumerate(ix):
                        for b, j in enumerate(iy):
                            data[cc(i,j)] = next(viter)

                else:
                    for a, i in enumerate(ix):
                        for b, j in enumerate(iy):
                            data[cc(i,j)] = value

            elif len(indexer) == 1:
                data[cc(*indexer)] = value
            else:
                raise NotImplementedError
        else:
            data[cc(*indexer)] = value
    else:
        data[cc(*indexer)] = value
