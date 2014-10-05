from __future__ import absolute_import, division, print_function

from datashape import dshape

from .core import Expr
from .arithmetic import UnaryOp

# Here follows a large number of unary operators.  These were selected by
# taking the intersection of the functions in ``math`` and ``numpy``

class RealMath(UnaryOp):
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


class IntegerMath(UnaryOp):
    """ Mathematical unary operator with int valued dshape like ceil, floor """
    @property
    def dshape(self):
        return dshape('int')


class ceil(IntegerMath): pass
class floor(IntegerMath): pass
class trunc(IntegerMath): pass


class BooleanMath(UnaryOp):
    """ Mathematical unary operator with bool valued dshape like isnan """
    @property
    def dshape(self):
        return dshape('bool')


class isnan(BooleanMath): pass
