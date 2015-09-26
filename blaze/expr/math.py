from __future__ import absolute_import, division, print_function

from datashape import Option, real, int_, bool_, isreal, isnumeric

from .arithmetic import UnaryOp, BinOp
from .expressions import schema_method_list


# Here follows a large number of unary operators.  These were selected by
# taking the intersection of the functions in ``math`` and ``numpy``

__all__ = ['abs', 'sqrt', 'sin', 'sinh', 'cos', 'cosh', 'tan', 'tanh', 'exp',
           'expm1', 'log', 'log10', 'log1p', 'acos', 'acosh', 'asin', 'asinh',
           'atan', 'atanh', 'radians', 'degrees', 'ceil', 'floor', 'trunc',
           'isnan', 'notnull', 'RealMath', 'IntegerMath', 'BooleanMath',
           'Math']


class UnaryMath(UnaryOp):

    """Mathematical unary operator with real valued dshape like sin, or exp
    """
    __slots__ = '_hash', '_child'

    _dtype = real
    _dtype = real



class abs(UnaryMath):
    pass


class sqrt(UnaryMath):
    pass


class sin(UnaryMath):
    pass


class sinh(UnaryMath):
    pass


class cos(UnaryMath):
    pass


class cosh(UnaryMath):
    pass


class tan(UnaryMath):
    pass


class tanh(UnaryMath):
    pass


class exp(UnaryMath):
    pass


class expm1(UnaryMath):
    pass


class log(UnaryMath):
    pass


class log10(UnaryMath):
    pass


class log1p(UnaryMath):
    pass


class acos(UnaryMath):
    pass


class acosh(UnaryMath):
    pass


class asin(UnaryMath):
    pass


class asinh(UnaryMath):
    pass


class atan(UnaryMath):
    pass


class atanh(UnaryMath):
    pass


class radians(UnaryMath):
    pass


class degrees(UnaryMath):
    pass


    """ Mathematical unary operator with int valued dshape like ceil, floor """
    _dtype = int_


class ceil(UnaryIntegerMath):
    pass


class floor(UnaryIntegerMath):
    pass


class trunc(UnaryIntegerMath):
    pass


class isnan(UnaryMath):
    _dtype = bool_


class notnull(UnaryOp):

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
    _dtype = bool_


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
