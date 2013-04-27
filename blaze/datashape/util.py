#------------------------------------------------------------------------
# Utility Functions for DataShapes
#------------------------------------------------------------------------

def cat_dshapes(dslist):
    """
    Concatenates a list of dshapes together along
    the first axis. Raises an error if there is
    a mismatch along another axis or the measures
    are different.

    Requires that the leading dimension be a known
    size for all data shapes.
    TODO: Relax this restriction to support
          streaming dimensions.
    """
    if len(dslist) == 0:
        raise ValueError('Cannot concatenate an empty list of dshapes')
    elif len(dslist) == 1:
        return dslist[0]

    outer_dim_size = operator.index(dslist[0][0])
    inner_ds = dslist[0][1:]
    for ds in dslist[1:]:
        outer_dim_size += operator.index(ds[0])
        if ds[1:] != inner_ds:
            raise ValueError(('The datashapes to concatenate much all match after'
                            ' the first dimension (%s vs %s)') %
                            (inner_ds, ds[1:]))
    return DataShape([Fixed(outer_dim_size)] + list(inner_ds))


def broadcastable(dslist, ranks=None, rankconnect=[]):
    """Return output (outer) shape if datashapes are broadcastable.

    The default is to assume broadcasting over a scalar operation.  If the kernel 
    to be applied takes arrays as arguments, then rank and rank-connect provide the
    inner-shape information with ranks a list of integers indicating the kernel rank
    required for each argument and rank-connect a list of sets of tuples where each
    tuple is a (argument, inner-dim) 
    """
    if ranks is None:
        ranks = [0]*len(dslist)

    shapes = [dshape.shape for dshape in dslist]
    splitshapes = [(shape[:len(shape)-rank], shape[len(shape)-rank:])
                             for shape, rank in zip(shapes, ranks)]
    outshapes, inshapes = zip(*splitshapes)

    # broadcast outer-dimensions
    maxshape = max(len(shape) for shape in outshapes)
    outshapes = [(1,)*(maxshape-len(shape))+shape for shape in outshapes]
    for shape1, shape2 in itertools.combinations(outshapes, 2):
        if any((dim1 != 1 and dim2 != 1 and dim1 != dim2) 
                  for dim1, dim2 in zip(shape1, shape2)):
            raise TypeError("Outer-dimensions are not broadcastable to the same shape")


def test_cat_dshapes():
    pass

def test_broadcastable():
    dslist = []

def test():
    test_cat_dshapes()
    test_broadcastable()

if __name__ == '__main__':
    test()


