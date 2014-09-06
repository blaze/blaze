""" An abstract array

>>> x = ArraySymbol('x', '5 * 3 * float32')
>>> x.sum(axis=0)
"""
from __future__ import absolute_import, division, print_function
from datashape import dshape, DataShape, Record, isdimension, Option
from datashape import coretypes as ct
import datashape

from . import scalar
from .core import Expr, path
from .scalar import ScalarSymbol, Number, Scalar
from .scalar import (Eq, Ne, Lt, Le, Gt, Ge, Add, Mult, Div, Sub, Pow, Mod, Or,
                     And, USub, Not, eval_str, FloorDiv, NumberInterface)
from ..compatibility import _strtypes, builtins, unicode, basestring, map, zip
from ..dispatch import dispatch


class ArrayExpr(Expr):
    __inputs__ = 'child',

    def _len(self):
        try:
            return int(self.dshape[0])
        except TypeError:
            raise ValueError('Can not determine length of table with the '
                    'following datashape: %s' % self.dshape)

    def __len__(self):
        return self._len()

    def __nonzero__(self):
        return True

    def __bool__(self):
        return True

    @property
    def shape(self):
        return datashape.to_numpy(self.dshape)[0]

    @property
    def dtype(self):
        return datashape.to_numpy_dtype(self.schema)

    @property
    def schema(self):
        return dshape(self.dshape[-1])

    @property
    def ndim(self):
        return len(self.dshape) - 1

    def __dir__(self):
        return sorted(set(dir(type(self)) + list(self.__dict__.keys()) +
                          self.names))

    def __getitem__(self, key):
        if (    isinstance(key, tuple)
            and len(key) == self.ndim
            and all(isinstance(k, (int, Scalar)) for k in key)):
            return ArrayElement(self, key)
        elif isinstance(key, tuple):
            return Slice(self, key)

    def sum(self, axis=None):
        return sum(self, axis)

    def min(self, axis=None):
        return min(self, axis)

    def max(self, axis=None):
        return max(self, axis)

    def mean(self, axis=None):
        return mean(self, axis)

    def var(self, axis=None):
        return var(self, axis)

    def std(self, axis=None):
        return std(self, axis)

    def any(self, axis=None):
        return any(self, axis)

    def all(self, axis=None):
        return all(self, axis)


class ArraySymbol(ArrayExpr):
    __slots__ = '_name', 'dshape'

    def __init__(self, name, dshape):
        self._name = name
        self.dshape = datashape.dshape(dshape)

    def __str__(self):
        return self._name


class ArrayElement(Scalar):
    __slots__ = 'child', 'index'

    @property
    def dshape(self):
        return self.child.dshape.subshape[self.index]


class Slice(ArrayExpr):
    __slots__ = 'child', 'index'

    @property
    def dshape(self):
        return self.child.dshape.subshape[self.index]

class Reduction(Expr):
    __slots__ = 'child', 'axis'

    @property
    def dshape(self):
        if self.axis == None:
            return self.child.schema
        if isinstance(self.axis, int):
            axes = [self.axis]
        else:
            axes = self.axis
        s = tuple(slice(None) if i not in axes else 0
                            for i in range(self.child.ndim))
        return self.child.dshape.subshape[s]

class sum(Reduction): pass
class min(Reduction): pass
class max(Reduction): pass
class mean(Reduction): pass
class std(Reduction): pass
class var(Reduction): pass
class any(Reduction): pass
class all(Reduction): pass
