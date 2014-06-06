
from blaze.expr.core import Expr
from datashape import dshape

class Scalar(Expr):
    pass


class BinOp(Scalar):
    __slots__ = 'lhs', 'rhs'

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return '%s %s %s' % (self.lhs, self.symbol, self.rhs)


class UnaryOp(Scalar):
    __slots__ = 'parent',

    def __init__(self, table):
        self.parent = table

    def __str__(self):
        return '%s(%s)' % (self.symbol, self.parent)

    @property
    def symbol(self):
        return type(self).__name__


class ScalarSymbol(Scalar):
    __slots__ = 'token', 'dtype'

    def __init__(self, token, dtype):
        self.token = token
        self.dtype = dtype

    @property
    def dshape(self):
        return dshape(self.dtype)
