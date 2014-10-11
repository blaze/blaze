from datashape import Option, real, dshape, Record
from datashape.predicates import iscollection
from .expressions import ElemWise, Expr


__all__ = ['IsNull', 'isnull', 'DropNA', 'dropna']


class IsNull(ElemWise):
    __slots__ = '_child',

    @property
    def schema(self):
        return dshape('bool')


def isnull(expr):
    return IsNull(expr)


class DropNA(Expr):
    __slots__ = '_child', 'how'

    @property
    def dshape(self):
        return self._child.dshape


def dropna(expr, how='any'):
    return DropNA(expr, how=how)


from .expressions import schema_method_list, dshape_method_list


schema_method_list.extend([
    (lambda ds: isinstance(ds, (Option, Record)) or ds == real, set([isnull]))
])


def isdropna(ds):
    return (isinstance(ds.measure, (Option, Record)) or ds.measure == real) and iscollection(ds)


dshape_method_list.extend([
    (isdropna, set([dropna]))
])
