from __future__ import absolute_import, division, print_function

import operator
from toolz import first
import numpy as np
from datashape import dshape
from dateutil.parser import parse as dt_parse
from datashape.predicates import isscalar
from datashape import coretypes as ct

from .core import parenthesize, eval_str
from .expressions import Expr
from ..dispatch import dispatch
from ..compatibility import _strtypes


__all__ = '''BinOp UnaryOp Arithmetic Add Mult Sub Div FloorDiv Pow Mod USub
Relational Eq Ne Ge Lt Le Gt Gt And Or Not'''.split()

@dispatch(ct.Option, object)
def scalar_coerce(ds, val):
    if val or val == 0:
        return scalar_coerce(ds.ty, val)
    else:
        return None

@dispatch(ct.Date, _strtypes)
def scalar_coerce(_, val):
    dt = dt_parse(val)
    if dt.time():
        raise ValueError("Can not coerce %s to type Date, "
                "contains time information")
    return dt.date()

@dispatch(ct.DateTime, _strtypes)
def scalar_coerce(_, val):
    return dt_parse(val)

@dispatch(ct.CType, _strtypes)
def scalar_coerce(dt, val):
    return np.asscalar(np.asarray(val, dtype=dt.to_numpy_dtype()))

@dispatch(ct.Record, object)
def scalar_coerce(rec, val):
    if len(rec.fields) == 1:
        return scalar_coerce(first(rec.types), val)
    else:
        raise TypeError("Trying to coerce complex datashape\n"
                "got dshape: %s\n"
                "scalar_coerce only intended for scalar values" % rec)

@dispatch(ct.DataShape, object)
def scalar_coerce(ds, val):
    if len(ds) == 1:
        return scalar_coerce(ds[0], val)
    else:
        raise TypeError("Trying to coerce dimensional datashape\n"
                "got dshape: %s\n"
                "scalar_coerce only intended for scalar values" % ds)

@dispatch(object, object)
def scalar_coerce(dtype, val):
    return val

@dispatch(_strtypes, object)
def scalar_coerce(ds, val):
    return scalar_coerce(dshape(ds), val)


class BinOp(Expr):
    __slots__ = 'lhs', 'rhs'
    __inputs__ = 'lhs', 'rhs'

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        lhs = parenthesize(eval_str(self.lhs))
        rhs = parenthesize(eval_str(self.rhs))
        return '%s %s %s' % (lhs, self.symbol, rhs)


class UnaryOp(Expr):
    __slots__ = '_child',

    def __init__(self, child):
        self._child = child

    def __str__(self):
        return '%s(%s)' % (self.symbol, eval_str(self._child))

    @property
    def symbol(self):
        return type(self).__name__


class Arithmetic(BinOp):
    """ Super class for arithmetic operators like add or mul """
    _dtype = 'real'
    @property
    def dshape(self):
        # TODO: better inference.  e.g. int + int -> int
        return dshape(self._dtype)


class Add(Arithmetic):
    symbol = '+'
    op = operator.add


class Mult(Arithmetic):
    symbol = '*'
    op = operator.mul


class Sub(Arithmetic):
    symbol = '-'
    op = operator.sub


class Div(Arithmetic):
    symbol = '/'
    op = operator.truediv


class FloorDiv(Arithmetic):
    symbol = '//'
    op = operator.floordiv


class Pow(Arithmetic):
    symbol = '**'
    op = operator.pow


class Mod(Arithmetic):
    symbol = '%'
    op = operator.mod


class USub(UnaryOp):
    op = operator.neg

    def __str__(self):
        return '-%s' % self._child

    @property
    def dshape(self):
        # TODO: better inference.  -uint -> int
        return self._child.dshape


def _neg(self):
    return USub(self)

def _add(self, other):
    return Add(self, scalar_coerce(self.dshape, other))

def _radd(self, other):
    return Add(scalar_coerce(self.dshape, other), self)

def _mul(self, other):
    return Mult(self, scalar_coerce(self.dshape, other))

def _rmul(self, other):
    return Mult(scalar_coerce(self.dshape, other), self)

def _div(self, other):
    return Div(self, scalar_coerce(self.dshape, other))

def _rdiv(self, other):
    return Div(scalar_coerce(self.dshape, other), self)

def _floordiv(self, other):
    return FloorDiv(self, scalar_coerce(self.dshape, other))

def _rfloordiv(self, other):
    return FloorDiv(scalar_coerce(self.dshape, other), self)

def _sub(self, other):
    return Sub(self, scalar_coerce(self.dshape, other))

def _rsub(self, other):
    return Sub(scalar_coerce(self.dshape, other), self)

def _pow(self, other):
    return Pow(self, scalar_coerce(self.dshape, other))

def _rpow(self, other):
    return Pow(scalar_coerce(self.dshape, other), self)

def _mod(self, other):
    return Mod(self, scalar_coerce(self.dshape, other))

def _rmod(self, other):
    return Mod(scalar_coerce(self.dshape, other), self)


class Relational(BinOp):
    @property
    def dshape(self):
        return dshape('bool')


class Eq(Relational):
    symbol = '=='
    op = operator.eq


class Ne(Relational):
    symbol = '!='
    op = operator.ne


class Ge(Relational):
    symbol = '>='
    op = operator.ge


class Le(Relational):
    symbol = '<='
    op = operator.le


class Gt(Relational):
    symbol = '>'
    op = operator.gt


class Lt(Relational):
    symbol = '<'
    op = operator.lt


class And(Arithmetic):
    symbol = '&'
    op = operator.and_
    _dtype = 'bool'


class Or(Arithmetic):
    symbol = '|'
    op = operator.or_
    _dtype = 'bool'


class Not(UnaryOp):
    symbol = '~'
    op = operator.invert

    @property
    def dshape(self):
        return dshape('bool')


def _eq(self, other):
    return Eq(self, scalar_coerce(self.dshape, other))

def _ne(self, other):
    return Ne(self, scalar_coerce(self.dshape, other))

def _lt(self, other):
    return Lt(self, scalar_coerce(self.dshape, other))

def _le(self, other):
    return Le(self, scalar_coerce(self.dshape, other))

def _gt(self, other):
    return Gt(self, scalar_coerce(self.dshape, other))

def _ge(self, other):
    return Ge(self, scalar_coerce(self.dshape, other))

def _invert(self):
    return Invert(self)

def _and(self, other):
    return And(self, other)

def _rand(self, other):
    return And(other, self)

def _or(self, other):
    return Or(self, other)

def _ror(self, other):
    return Or(other, self)

def _invert(self):
    return Not(self)

Invert = Not
BitAnd = And
BitOr = Or


from .expressions import dshape_method_list

dshape_method_list.extend([
    (isscalar,
            set([_add, _radd, _mul,
            _rmul, _div, _rdiv, _floordiv, _rfloordiv, _sub, _rsub, _pow,
            _rpow, _mod, _rmod,  _neg])),
    (isscalar, set([_eq, _ne, _lt, _le, _gt, _ge])),
    (isscalar, set([_or, _ror, _and, _rand, _invert])),
    ])
