import datashape


def istabular(expr):
    return datashape.istabular(expr.dshape)

def isscalar(expr):
    return datashape.isscalar(expr.dshape)

def iscolumn(expr):
    if hasattr(expr, 'iscolumn'):
        return expr.iscolumn
    if (len(expr.dshape.shape) != 1
            or not datashape.isscalar(expr.dshape.measure)):
        return False
    if (isinstance(expr.dshape.measure, datashape.Record)
            and len(expr.dshape.measure.names) > 1):
        return False
    # TODO: Still ambiguous about record schemas with len 1
