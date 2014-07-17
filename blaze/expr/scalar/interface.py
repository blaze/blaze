from __future__ import absolute_import, division, print_function

from datashape import dshape
from toolz import merge

from ..core import Expr
from .core import Scalar
from .numbers import NumberInterface
from . import numbers
from .boolean import BooleanInterface

class ScalarSymbol(NumberInterface, BooleanInterface):
    __slots__ = '_name', 'dtype'

    def __init__(self, name, dtype='real'):
        self._name = name
        self.dtype = dtype

    @property
    def dshape(self):
        return dshape(self.dtype)

    def __str__(self):
        return str(self._name)

    __hash__ = Expr.__hash__


safe_scope = {'__builtins__': {},  # Python 2
              'builtins': {}}      # Python 3
# Operations like sin, cos, exp, isnan, floor, ceil, ...
math_operators = dict((k, v) for k, v in numbers.__dict__.items()
                      if isinstance(v, type) and issubclass(v, Scalar))

# Cripple fancy attempts
illegal_terms = '__', 'lambda', 'for', 'if', ':', '.'

def exprify(expr, dtypes):
    """ Transform string into scalar expression

    >>> expr = exprify('x + y', {'x': 'int64', 'y': 'real'})
    >>> expr
    x + y
    >>> isinstance(expr, Expr)
    True
    >>> expr.lhs.dshape
    dshape("int64")
    """
    if any(term in expr for term in illegal_terms):
        raise ValueError('Unclean input' % expr)

    variables = dict((k, ScalarSymbol(k, v)) for k, v in dtypes.items())

    scope = merge(safe_scope, math_operators, variables)

    return eval(expr, scope)
