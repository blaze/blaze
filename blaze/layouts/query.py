import numpy as np

#------------------------------------------------------------------------
# Manifest Data Retrieval
#------------------------------------------------------------------------

def retrieve(cc, indexer):
    """
    Top dispatcher for data retrieval performs the checks to
    determine what the user passed in to the brackets as an
    indexer.

    Handles the various objects that can be passed into brackets
    on an Indexable.

        * [1]
        * [1,2]
        * [1:10:2]
        * [(1:10:2, 1:10:2)]
        * ['foo']
        * ['foo', 1:10]

    Parameters
    ----------

        :cc: Coordinate transform function

        :indexer: Outer indexer passed from user.

        :data: The byte provider to read bytes (if ``MANIFEST``) or
               schedule a read operation ( if ``DEFERRED``).

    """

    if isinstance(indexer, int):
        return getitem(cc, indexer)
    elif isinstance(indexer, slice):
        return getitem(cc, indexer)
    elif isinstance(indexer, tuple):
        if len(indexer) == 1:
            return getitem(cc, indexer)
        elif len(indexer) == 2:
            return getslice(cc, indexer)
        else:
            raise NotImplementedError
    elif isinstance(indexer, tuple):
        return getlabel(cc, indexer)
    else:
        raise NotImplementedError

def getslice(cc, indexer):
    # Shortcut for accessing data in carray container
    # (this needs to be more general?)
    elt, lc = cc(indexer)
    return elt.ca[lc]

    # a = indexer[0]
    # b = indexer[1]

    # max1 = data.bounds[0]
    # max2 = data.bounds[1]

    # ix = range(a.start or 0, a.stop or max1, a.step or 1)
    # iy = range(a.start or 0, b.stop or max2, b.step or 1)

    # # TODO: use source.empty() to generalize
    # res = np.empty((len(ix), len(iy)))

    # for a, i in enumerate(ix):
    #     for b, j in enumerate(iy):
    #         elt, lc = cc(indexer)
    #         res[a,b] = elt[cc(i,j)]
    # return res

def getitem(cc, indexer):
    # local coordinates
    elt, lc = cc(indexer)

    # Read from the generic interface in terms of the local
    # coordinates.
    datum = elt.read(elt, lc)
    res = np.array(datum)
    return res

def getlabel(cc, indexer):
    pass

#------------------------------------------------------------------------
# Manifest Data Write
#------------------------------------------------------------------------

def write(cc, indexer, value):
    # Shortcut for accessing data in carray container
    # (this needs to be more general?)
    elt, lc = cc(indexer)
    elt.ca[lc] = value


#------------------------------------------------------------------------
# Deferred Data Access
#------------------------------------------------------------------------
