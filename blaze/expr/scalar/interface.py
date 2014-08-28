from __future__ import absolute_import, division, print_function

from ..core import Expr
from datashape import dshape
from .boolean import BooleanInterface
from .numbers import NumberInterface


class ScalarSymbol(NumberInterface, BooleanInterface):
    __slots__ = '_name', 'dtype'
    __inputs__ = ()

    def __init__(self, name, dtype='real'):
        self._name = name
        self.dtype = dshape(dtype)

    @property
    def name(self):
        return self._name

    @property
    def dshape(self):
        return dshape(self.dtype)

    def __str__(self):
        return str(self._name)

    __hash__ = Expr.__hash__
