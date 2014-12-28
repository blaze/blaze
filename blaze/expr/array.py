from __future__ import absolute_import, division, print_function

from .expressions import Expr, ndim
from datashape import DataShape
from collections import Iterable

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


class TensorDot(Expr):
    __slots__ = '_hash', 'lhs', 'rhs', '_left_axes', '_right_axes'
    __inputs__ = 'lhs', 'rhs'

    @property
    def dshape(self):
        shape = tuple([d for i, d in enumerate(self.lhs.shape)
                         if i not in self._left_axes] +
                      [d for i, d in enumerate(self.rhs.shape)
                         if i not in self._right_axes])
        # TODO: handle type promotion
        return DataShape(*(shape + (self.lhs.dshape.measure,)))

def tensordot(lhs, rhs, axes=None):
    if axes is None:
        if ndim(lhs) == 2:
            left = ndim(lhs) - 1
        right = 0
    elif isinstance(axes, Iterable):
        left, right = axes
    else:
        left, right = axes, axes

    if isinstance(left, int):
        left = (left,)
    if isinstance(right, int):
        right = (right,)
    if isinstance(left, list):
        left = tuple(left)
    if isinstance(right, list):
        right = tuple(right)

    return TensorDot(lhs, rhs, left, right)

def dot(lhs, rhs):
    return tensordot(lhs, rhs)


from datashape.predicates import isnumeric, isboolean
from .expressions import dshape_method_list, method_properties

dshape_method_list.extend([
    (lambda ds: ndim(ds) > 1, set([transpose])),
    (lambda ds: ndim(ds) == 2, set([T])),
    (lambda ds: ndim(ds) >= 1 and (isnumeric(ds) or isboolean(ds)), set([dot]))
    ])

method_properties.add(T)
