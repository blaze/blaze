from __future__ import absolute_import

__all__ = ['dopen', 'dshape', 'cat_dshapes', 'broadcastable']
            
import operator
from .coretypes import DataShape, Fixed

#------------------------------------------------------------------------
# Utility Functions for DataShapes
#------------------------------------------------------------------------

def dopen(fname):
    contents = open(fname).read()
    return parser.parse_extern(contents)

def dshape(o):
    if isinstance(o, str):
        return parser.parse(o)
    elif isinstance(o, DataShape):
        return o
    else:
        raise TypeError('Cannot create dshape from object of type %s' % type(o))

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

    The default is to assume broadcasting over a scalar operation.  
    However, if the kernel to be applied takes arrays as arguments, 
    then rank and rank-connect provide the inner-shape information with 
    ranks a list of integers indicating the kernel rank of each argument
    and rank-connect a list of sets of tuples where each set contains the 
    dimensions that must match and each tuple is (argument #, inner-dim #)
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

    outshape = tuple(map(max, zip(*outshapes)))

    for connect in rankconnect:
        for (arg1, dim1), (arg2,dim2) in itertools.combinations(connect, 2):
            if (inshapes[arg1][dim1] != inshapes[arg2][dim2]):
                raise TypeError("Inner dimensions do not match in " + 
                                "argument %d and argument %d" % (arg1, arg2))

    return outshape


def test_cat_dshapes():
    pass

def test_broadcastable():
    from blaze.datashape import dshape
    dslist = [dshape('10,20,30,int32'), dshape('20,30,int32'), dshape('int32')]
    outshape = broadcastable(dslist, ranks=[1,1,0])
    assert outshape == (10,20)

def test():
    test_cat_dshapes()
    test_broadcastable()

if __name__ == '__main__':
    test()


