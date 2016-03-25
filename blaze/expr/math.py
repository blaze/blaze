from __future__ import absolute_import, division, print_function

from datashape import Option, real, int_, bool_, isreal, isnumeric

from .arithmetic import UnaryOp, BinOp, Arithmetic
from .expressions import schema_method_list
from ..compatibility import builtins


# Here follows a large number of unary operators.  These were selected by
# taking the intersection of the functions in ``math`` and ``numpy``

__all__ = ['abs', 'sqrt', 'sin', 'sinh', 'cos', 'cosh', 'tan', 'tanh', 'exp',
           'expm1', 'log', 'log10', 'log1p', 'acos', 'acosh', 'asin', 'asinh',
           'atan', 'atanh', 'radians', 'degrees', 'atan2', 'ceil', 'floor',
           'trunc', 'isnan', 'notnull', 'UnaryMath', 'BinaryMath',
           'greatest', 'least']


class UnaryMath(UnaryOp):

    """Mathematical unary operator with real valued dshape like sin, or exp
    """
    __slots__ = '_hash', '_child'

    _dtype = real


class BinaryMath(BinOp):
    __slots__ = '_hash', 'lhs', 'rhs'
    __inputs__ = 'lhs', 'rhs'

    _dtype = real

    def __str__(self):
        return '%s(%s, %s)' % (type(self).__name__, self.lhs, self.rhs)


_unary_math_names = (
    'abs',
    'sqrt',
    'sin',
    'sinh',
    'cos',
    'cosh',
    'tan',
    'tanh',
    'exp',
    'expm1',
    'log',
    'log10',
    'log1p',
    'acos',
    'acosh',
    'asin',
    'asinh',
    'atan',
    'atanh',
    'radians',
    'degrees',
)


for name in _unary_math_names:
    locals()[name] = type(name, (UnaryMath,), {})


_binary_math_names = (
    'atan2',
    'copysign',
    'fmod',
    'hypot',
    'ldexp',
)

for name in _binary_math_names:
    locals()[name] = type(name, (BinaryMath,), {})


class greatest(Arithmetic):
    __slots__ = '_hash', 'lhs', 'rhs'
    __inputs__ = 'lhs', 'rhs'
    op = builtins.max

    def __str__(self):
        return 'greatest(%s, %s)' % (self.lhs, self.rhs)


class least(Arithmetic):
    __slots__ = '_hash', 'lhs', 'rhs'
    __inputs__ = 'lhs', 'rhs'
    op = builtins.min

    def __str__(self):
        return 'least(%s, %s)' % (self.lhs, self.rhs)


_unary_integer_math = (
    'ceil',
    'floor',
    'trunc',
)

for name in _unary_integer_math:
    locals()[name] = type(name, (UnaryMath,), dict(_dtype=int_))


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
