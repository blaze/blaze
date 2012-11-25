from numpy import np

def getitem(cc, data, indexer):

    if isinstance(indexer, tuple):
        if len(indexer) == 2:
            idx0 = indexer[0]
            idx1 = indexer[1]

            ix = range(idx0.start, idx0.stop, idx0.step or 1)
            iy = range(idx1.start, idx1.stop, idx1.step or 1)

            # TODO: use source.empty() to generalize
            res = np.empty((len(ix), len(iy)))

            for a, i in enumerate(ix):
                for b, j in enumerate(iy):
                    res[a,b] = data[cc(i,j)]
            return res

        elif len(indexer) == 1:
            return data[cc(*indexer)]
        else:
            raise NotImplementedError

def setitem(cc, data, indexer, value):

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
