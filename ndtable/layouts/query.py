import numpy as np

#------------------------------------------------------------------------
# Manifest Data Retrieval
#------------------------------------------------------------------------

def retrieve(cc, indexer, data):
    """
    Top dispatcher for data retrieval performs the checks to
    determine what the user passed in to the brackets as an
    indexer.

    Parameters
    ----------

        :cc: Coordinate transform function

        :indexer: Outer indexer passed from user.

        :data: The byte provider to read bytes (if ``MANIFEST``) or
               schedule a read operation ( if ``DEFERRED``).

    """

    if isinstance(indexer, tuple):
        if len(indexer) == 1:
            return getitem(cc, indexer, data)
        elif len(indexer) == 2:
            return getslice(cc, indexer, data)
        else:
            raise NotImplementedError
    elif isinstance(indexer, tuple):
        return getlabel(cc, indexer, data)
    else:
        raise NotImplementedError

def getslice(cc, indexer, data):
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

def getitem(cc, indexer, data):
    datum = data[cc(indexer)]
    res = np.array(datum)
    return res

def getlabel(cc, indexer, data):
    pass

#------------------------------------------------------------------------
# Manifest Data Write
#------------------------------------------------------------------------

def write(cc, indexer, data, value):

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

#------------------------------------------------------------------------
# Deferred Data Access
#------------------------------------------------------------------------
