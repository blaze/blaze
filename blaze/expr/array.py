from .expressions import Expr, ndim
from datashape import DataShape

class Transpose(Expr):
    __slots__ = '_hash', '_child', 'axes'

    @property
    def dshape(self):
        s = self._child.shape
        return DataShape(*(tuple([s[i] for i in self.axes]) +
                           (self._child.dshape.measure,)))


def transpose(expr, axes=None):
    if axes is None:
        assert ndim(expr) == 2
        axes = (1, 0)
    return Transpose(expr, axes)


def T(expr):
    return transpose(expr)


from .expressions import dshape_method_list, method_properties

dshape_method_list.extend([
    (lambda ds: ndim(ds) > 1, set([transpose])),
    (lambda ds: ndim(ds) == 2, set([T])),
    ])

method_properties.add(T)
