import operator
from datashape import dshape
from .core import Scalar, BinOp, UnaryOp


class Boolean(Scalar):
    @property
    def dshape(self):
        return dshape('bool')

    def __not__(self):
        return Not(self)

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)


class Relational(BinOp, Boolean):
    pass


class Eq(Relational):
    symbol = '=='
    op = operator.eq


class GT(Relational):
    symbol = '>'
    op = operator.gt


class LT(Relational):
    symbol = '<'
    op = operator.lt


class And(Boolean):
    symbol = '&'
    op = operator.and_


class Or(Boolean):
    symbol = '|'
    op = operator.or_
