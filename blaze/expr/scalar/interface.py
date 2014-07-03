from datashape import dshape
from toolz import merge

from ..core import Expr
from .core import Scalar
from .numbers import NumberInterface
from . import numbers
from .boolean import BooleanInterface

class ScalarSymbol(NumberInterface, BooleanInterface):
    __slots__ = 'name', 'dtype'

    def __init__(self, name, dtype='real'):
        self.name = name
        self.dtype = dtype

    @property
    def dshape(self):
        return dshape(self.dtype)

    def __str__(self):
        return str(self.name)

    __hash__ = Expr.__hash__


safe_scope = {'__builtins__': {}}
math_operators = dict((k, v) for k, v in numbers.__dict__.items()
                      if isinstance(v, type) and issubclass(v, Scalar))


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
    if '__' in expr:
        raise ValueError('Unclean input' % expr)
    variables = dict((k, ScalarSymbol(k, v)) for k, v in dtypes.items())

    d = merge(safe_scope, math_operators, variables)

    return eval(expr, d)
