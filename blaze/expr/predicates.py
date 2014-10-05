from __future__ import absolute_import, division, print_function

import datashape
from datashape import dshape, DataShape, Option, Unit
from datashape.predicates import isscalar


def istabular(expr):
    return datashape.istabular(expr.dshape)

def iscolumn(expr):
    return len(expr.dshape.shape) == 1 and isscalar(expr.dshape.measure)
