from __future__ import absolute_import, division, print_function

from datashape import Record
from blaze.expr.core import Expr


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

    @property
    def name(self):
        try:
            return self.dshape[0].names[0]
        except:
            raise ValueError("No name found in dshape %s" % self.dshape)

    @property
    def dtype(self):
        if isinstance(self.dshape[0], Record):
            return self.dshape[0].types[0]
        else:
            return self.dshape


class BinOp(Scalar):
    __slots__ = 'lhs', 'rhs'
    __inputs__ = 'lhs', 'rhs'

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        lhs = parenthesize(eval_str(self.lhs))
        rhs = parenthesize(eval_str(self.rhs))
        return '%s %s %s' % (lhs, self.symbol, rhs)


class UnaryOp(Scalar):
    __slots__ = 'child',

    def __init__(self, child):
        self.child = child

    def __str__(self):
        return '%s(%s)' % (self.symbol, eval_str(self.child))

    @property
    def symbol(self):
        return type(self).__name__
