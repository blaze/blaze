import datashape
from datashape import dshape, DataShape, Option, Unit


def istabular(expr):
    return datashape.istabular(expr.dshape)

def isscalar(expr):
    return datashape.isscalar(expr.dshape)


def isunit(ds):
    """ Is this dshape a single dtype?

    >>> isunit('int')
    True
    >>> isunit('?int')
    True
    >>> isunit('{name: string, amount: int}')
    False
    """
    if isinstance(ds, str):
        ds = dshape(ds)
    if isinstance(ds, DataShape) and len(ds) == 1:
        ds = ds[0]
    if isinstance(ds, Option):
        ds = ds.ty
    return isinstance(ds, Unit)


def iscolumn(expr):
    return len(expr.dshape.shape) == 1 and isunit(expr.dshape.measure)
