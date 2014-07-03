from __future__ import absolute_import, division, print_function

import operator
from datashape import dshape
from dateutil.parser import parse as dt_parse
from .core import Scalar, BinOp, UnaryOp
from ..core import Expr
from ...dispatch import dispatch
from ...compatibility import _strtypes
from datashape import coretypes as ct
from .boolean import *


@dispatch(_strtypes, ct.Date)
def coerce(val, _):
    return dt_parse(val).date()

@dispatch(_strtypes, ct.DateTime)
def coerce(val, _):
    return dt_parse(val)

@dispatch(object, ct.DataShape)
def coerce(val, dtype):
    return coerce(val, dtype[0])

@dispatch(object, object)
def coerce(val, dtype):
    return val


class NumberInterface(Scalar):
    def __eq__(self, other):
        return Eq(self, coerce(other, self.dshape))

    def __ne__(self, other):
        return NE(self, coerce(other, self.dshape))

    def __lt__(self, other):
        return LT(self, coerce(other, self.dshape))

    def __le__(self, other):
        return LE(self, coerce(other, self.dshape))

    def __gt__(self, other):
        return GT(self, coerce(other, self.dshape))

    def __ge__(self, other):
        return GE(self, coerce(other, self.dshape))

    def __neg__(self):
        return Neg(self)

    def __add__(self, other):
        return Add(self, coerce(other, self.dshape))

    def __radd__(self, other):
        return Add(coerce(other, self.dshape), self)

    def __mul__(self, other):
        return Mul(self, coerce(other, self.dshape))

    def __rmul__(self, other):
        return Mul(coerce(other, self.dshape), self)

    def __div__(self, other):
        return Div(self, coerce(other, self.dshape))

    __truediv__ = __div__

    def __rdiv__(self, other):
        return Div(coerce(other, self.dshape), self)

    def __sub__(self, other):
        return Sub(self, coerce(other, self.dshape))

    def __rsub__(self, other):
        return Sub(coerce(other, self.dshape), self)

    def __pow__(self, other):
        return Pow(self, coerce(other, self.dshape))

    def __rpow__(self, other):
        return Pow(coerce(other, self.dshape), self)

    def __mod__(self, other):
        return Mod(self, coerce(other, self.dshape))

    def __rmod__(self, other):
        return Mod(coerce(other, self.dshape), self)


class Number(NumberInterface):
    __hash__ = Expr.__hash__


class Arithmetic(BinOp, Number):
    """ Super class for arithmetic operators like add or mul """
    @property
    def dshape(self):
        # TODO: better inference.  e.g. int + int -> int
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


class Neg(UnaryOp, Number):
    op = operator.neg

    def __str__(self):
        return '-%s' % self.parent

    @property
    def dshape(self):
        # TODO: better inference.  -uint -> int
        return self.parent.dshape



# Here follows a large number of unary operators.  These were selected by
# taking the intersection of the functions in ``math`` and ``numpy``

class RealMath(Number, UnaryOp):
    """ Mathematical unary operator with real valued dshape like sin, or exp """
    @property
    def dshape(self):
        return dshape('real')


class sqrt(RealMath): pass

class sin(RealMath): pass
class sinh(RealMath): pass
class cos(RealMath): pass
class cosh(RealMath): pass
class tan(RealMath): pass
class tanh(RealMath): pass

class exp(RealMath): pass
class expm1(RealMath): pass
class log(RealMath): pass
class log10(RealMath): pass
class log1p(RealMath): pass

class acos(RealMath): pass
class acosh(RealMath): pass
class asin(RealMath): pass
class asinh(RealMath): pass
class atan(RealMath): pass
class atanh(RealMath): pass

class radians(RealMath): pass
class degrees(RealMath): pass


class IntegerMath(Number, UnaryOp):
    """ Mathematical unary operator with int valued dshape like ceil, floor """
    @property
    def dshape(self):
        return dshape('int')


class ceil(IntegerMath): pass
class floor(IntegerMath): pass
class trunc(IntegerMath): pass


class BooleanMath(Number, UnaryOp):
    """ Mathematical unary operator with bool valued dshape like isnan """
    @property
    def dshape(self):
        return dshape('bool')


class isnan(BooleanMath): pass
