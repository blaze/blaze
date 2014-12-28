from datashape import Option, dshape, DataShape, Var
from datashape.predicates import iscollection
from .expressions import ElemWise, Expr


__all__ = ['IsNull', 'isnull', 'DropNA', 'dropna']


class IsNull(ElemWise):
    __slots__ = '_hash', '_child',

    @property
    def schema(self):
        return dshape('bool')

    @property
    def dshape(self):
        return DataShape(Var(), self.schema)

    def __str__(self):
        return '%s.isnull()' % self._child


def isnull(expr):
    return IsNull(expr)


class DropNA(Expr):
    __slots__ = '_hash', '_child', 'how'

    @property
    def dshape(self):
        ds = self._child.dshape
        return DataShape(Var(), ds.measure.ty)

    def __str__(self):
        return '%s.dropna(how=%r)' % (self._child, self.how)


def dropna(expr, how='any'):
    return DropNA(expr, how=how)


from .expressions import schema_method_list, dshape_method_list


schema_method_list.extend([
    (lambda ds: isinstance(ds, Option), set([isnull]))
])


dshape_method_list.extend([
    (lambda ds: isinstance(ds.measure, Option) and iscollection(ds),
     set([dropna])),
])
