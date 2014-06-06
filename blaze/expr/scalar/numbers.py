from __future__ import absolute_import

import operator
from datashape import dshape
from .core import Scalar, ScalarSymbol, BinOp, UnaryOp
from ..core import Expr
from .boolean import Eq, LT, GT


class Number(Scalar):
    def __eq__(self, other):
        return Eq(self, other)

    def __lt__(self, other):
        return LT(self, other)

    def __gt__(self, other):
        return GT(self, other)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    def __div__(self, other):
        return Div(self, other)

    def __rdiv__(self, other):
        return Div(other, self)

    def __sub_(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __pow__(self, other):
        return Pow(self, other)

    def __rpow__(self, other):
        return Pow(other, self)

    def __mod__(self, other):
        return Mod(self, other)

    def __rmod__(self, other):
        return Mod(other, self)

    __hash__ = Expr.__hash__


class NumberSymbol(ScalarSymbol, Number):
    __slots__ = 'name', 'dtype'

    def __init__(self, name, dtype=None):
        self.name = name
        self.dtype = dtype or 'real'


class Arithmetic(BinOp, Number):
    @property
    def dshape(self):
        return dshape('real')


class Add(Arithmetic):
    symbol = '+'
    op = operator.add


class Mul(Arithmetic):
    symbol = '*'
    op = operator.mul


class Sub(Arithmetic):
    symbol = '-'
    op = operator.sub


class Div(Arithmetic):
    symbol = '/'
    op = operator.truediv


class Pow(Arithmetic):
    symbol = '**'
    op = operator.pow


class Mod(Arithmetic):
    symbol = '%'
    op = operator.mod


class UnaryMath(Number, UnaryOp):
    pass


class sin(UnaryMath): pass
class cos(UnaryMath): pass
class tan(UnaryMath): pass
class exp(UnaryMath): pass
class log(UnaryMath): pass
