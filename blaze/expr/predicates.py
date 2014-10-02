import datashape
from datashape import dshape, DataShape, Option, Unit
from datashape.predicates import isunit


def istabular(expr):
    return datashape.istabular(expr.dshape)

def isscalar(expr):
    return datashape.isscalar(expr.dshape)

def iscolumn(expr):
    return len(expr.dshape.shape) == 1 and isunit(expr.dshape.measure)
