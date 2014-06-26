from ..core import Expr
from .core import Scalar
from .numbers import NumberInterface
from .boolean import BooleanInterface
from datashape import dshape

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
    locals().update({k: ScalarSymbol(k, v) for k, v in dtypes.items()})
    return eval(expr)
