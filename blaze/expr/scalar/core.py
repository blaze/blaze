
from blaze.expr.core import Expr
from datashape import dshape


def eval_str(expr):
    """ String suitable for evaluation """
    if hasattr(expr, 'eval_str'):
        return expr.eval_str()
    elif isinstance(expr, str):
        return "'%s'" % expr
    else:
        return str(expr)

def parenthesize(s):
    """

    >>> parenthesize('1')
    '1'
    >>> parenthesize('1 + 2')
    '(1 + 2)'
    """
    if ' ' in s:
        return '(%s)' % s
    else:
        return s

class Scalar(Expr):
    def eval_str(self):
        return str(self)


class BinOp(Scalar):
    __slots__ = 'lhs', 'rhs'

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        lhs = parenthesize(eval_str(self.lhs))
        rhs = parenthesize(eval_str(self.rhs))
        return '%s %s %s' % (lhs, self.symbol, rhs)
        return '%s %s %s' % (self.lhs, self.symbol, self.rhs)


class UnaryOp(Scalar):
    __slots__ = 'parent',

    def __init__(self, table):
        self.parent = table

    def __str__(self):
        return '%s(%s)' % (self.symbol, eval_str(self.parent))

    @property
    def symbol(self):
        return type(self).__name__


class ScalarSymbol(Scalar):
    __slots__ = 'name', 'dtype'

    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype

    @property
    def dshape(self):
        return dshape(self.dtype)

    def __str__(self):
        return str(self.name)
