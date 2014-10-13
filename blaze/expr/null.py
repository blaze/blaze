from datashape import Option, dshape, DataShape, Var
from datashape.predicates import iscollection
from .expressions import ElemWise, Expr


__all__ = ['IsNull', 'isnull', 'DropNA', 'dropna']


class IsNull(ElemWise):
    __slots__ = '_child',

    @property
    def schema(self):
        return dshape('bool')

    @property
    def dshape(self):
        return DataShape(Var(), self.schema.measure)


def isnull(expr):
    return IsNull(expr)


class DropNA(Expr):
    __slots__ = '_child', 'how'

    @property
    def dshape(self):
        ds = self._child.dshape
        return DataShape(Var(), ds.measure.ty)


def dropna(expr, how='any'):
    return DropNA(expr, how=how)


from .expressions import schema_method_list, dshape_method_list


schema_method_list.extend([
    (lambda ds: isinstance(ds, Option), set([isnull]))
])


def isdropna(ds):
    return isinstance(ds.measure, Option) and iscollection(ds)


dshape_method_list.extend([
    (isdropna, set([dropna]))
])
