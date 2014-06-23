import operator
from datashape import dshape
from .core import Scalar, BinOp, UnaryOp


class BooleanInterface(Scalar):
    def __not__(self):
        return Not(self)

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)


class Boolean(BooleanInterface):
    @property
    def dshape(self):
        return dshape('bool')


class Relational(BinOp, Boolean):
    pass


class Eq(Relational):
    symbol = '=='
    op = operator.eq


class NE(Relational):
    symbol = '!='
    op = operator.ne


class GE(Relational):
    symbol = '>='
    op = operator.ge


class LE(Relational):
    symbol = '<='
    op = operator.le


class GT(Relational):
    symbol = '>'
    op = operator.gt


class LT(Relational):
    symbol = '<'
    op = operator.lt


class And(BinOp, Boolean):
    symbol = '&'
    op = operator.and_


class Or(BinOp, Boolean):
    symbol = '|'
    op = operator.or_


class Not(UnaryOp, Boolean):
    op = operator.not_
