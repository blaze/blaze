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
from .scalar import ScalarSymbol, Number
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
    def names(self):
        return self.schema[0].names

    def __dir__(self):
        return sorted(set(dir(type(self)) + list(self.__dict__.keys()) +
                          self.names))

class ArraySymbol(ArrayExpr):
    __slots__ = '_name', 'dshape'

    def __init__(self, name, dshape):
        self._name = name
        self.dshape = datashape.dshape(dshape)

    def __str__(self):
        return self._name
