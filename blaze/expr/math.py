from __future__ import absolute_import, division, print_function

from datashape import Option, real, int_, bool_, isreal, isnumeric

from .arithmetic import UnaryOp
from .expressions import schema_method_list


# Here follows a large number of unary operators.  These were selected by
# taking the intersection of the functions in ``math`` and ``numpy``

__all__ = ['abs', 'sqrt', 'sin', 'sinh', 'cos', 'cosh', 'tan', 'tanh', 'exp',
           'expm1', 'log', 'log10', 'log1p', 'acos', 'acosh', 'asin', 'asinh',
           'atan', 'atanh', 'radians', 'degrees', 'ceil', 'floor', 'trunc',
           'isnan', 'notnull', 'RealMath', 'IntegerMath', 'BooleanMath',
           'Math']


class Math(UnaryOp):
    pass


class RealMath(Math):
    """Mathematical unary operator with real valued dshape like sin, or exp
    """
    _dtype = real


class abs(RealMath): pass
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


class IntegerMath(Math):
    """ Mathematical unary operator with int valued dshape like ceil, floor """
    _dtype = int_


class ceil(IntegerMath): pass
class floor(IntegerMath): pass
class trunc(IntegerMath): pass


class BooleanMath(Math):
    """ Mathematical unary operator with bool valued dshape like isnan """
    _dtype = bool_


class isnan(BooleanMath): pass


class notnull(BooleanMath):
    """ Return whether an expression is not null

    Examples
    --------
    >>> from blaze import symbol, compute
    >>> s = symbol('s', 'var * int64')
    >>> expr = notnull(s)
    >>> expr.dshape
    dshape("var * bool")
    >>> list(compute(expr, [1, 2, None, 3]))
    [True, True, False, True]
    """
    pass


def truncate(expr, precision):
    """ Truncate number to precision

    Examples
    --------

    >>> from blaze import symbol, compute
    >>> x = symbol('x', 'real')
    >>> compute(x.truncate(10), 123)
    120
    >>> compute(x.truncate(0.1), 3.1415)  # doctest: +SKIP
    3.1
    """
    return expr // precision * precision


schema_method_list.extend([
    (isreal, set([isnan])),
    (isnumeric, set([truncate])),
    (lambda ds: isinstance(ds, Option) or isinstance(ds.measure, Option),
     set([notnull]))
])
