from __future__ import absolute_import, division, print_function

import operator
from toolz import first
import numpy as np
from datashape import dshape, var, DataShape
from dateutil.parser import parse as dt_parse
from datashape.predicates import isscalar
from datashape import coretypes as ct

from .core import parenthesize, eval_str
from .expressions import Expr, shape, ElemWise
from ..dispatch import dispatch
from ..compatibility import _strtypes


__all__ = '''BinOp UnaryOp Arithmetic Add Mult Sub Div FloorDiv Pow Mod USub
Relational Eq Ne Ge Lt Le Gt Gt And Or Not'''.split()


def name(o):
    if hasattr(o, '_name'):
        return o._name
    else:
        return None

class BinOp(ElemWise):
    __slots__ = '_hash', 'lhs', 'rhs'
    __inputs__ = 'lhs', 'rhs'

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        lhs = parenthesize(eval_str(self.lhs))
        rhs = parenthesize(eval_str(self.rhs))
        return '%s %s %s' % (lhs, self.symbol, rhs)

    @property
    def _name(self):
        if not isscalar(self.dshape.measure):
            return None
        l, r = name(self.lhs), name(self.rhs)
        if l and not r:
            return l
        if r and not l:
            return r
        if l == r:
            return l

    @property
    def _inputs(self):
        result = []
        if isinstance(self.lhs, Expr):
            result.append(self.lhs)
        if isinstance(self.rhs, Expr):
            result.append(self.rhs)
        return tuple(result)


def maxvar(L):
    """

    >>> maxvar([1, 2, var])
    Var()

    >>> maxvar([1, 2, 3])
    3
    """
    if var in L:
        return var
    else:
        return max(L)


def maxshape(shapes):
    """

    >>> maxshape([(10, 1), (1, 10), ()])
    (10, 10)
    """
    shapes = [shape for shape in shapes if shape]
    if not shapes:
        return ()
    if len(set(map(len, shapes))) != 1:
        raise ValueError("Only support arithmetic on expressions with equal "
                "number of dimensions.")
    return tuple(map(maxvar, zip(*shapes)))


class UnaryOp(ElemWise):
    __slots__ = '_hash', '_child',

    def __init__(self, child):
        self._child = child

    def __str__(self):
        return '%s(%s)' % (self.symbol, eval_str(self._child))

    @property
    def symbol(self):
        return type(self).__name__

    @property
    def dshape(self):
        return DataShape(*(shape(self._child) + (self._dtype,)))

    @property
    def _name(self):
        return self._child._name


class Arithmetic(BinOp):
    """ Super class for arithmetic operators like add or mul """
    _dtype = ct.real

    @property
    def dshape(self):
        # TODO: better inference.  e.g. int + int -> int
        return DataShape(*(maxshape([shape(self.lhs), shape(self.rhs)]) + (self._dtype,)))


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
    symbol = '-'

    def __str__(self):
        return '-%s' % parenthesize(eval_str(self._child))

    @property
    def _dtype(self):
        # TODO: better inference.  -uint -> int
        return self._child.schema


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
    return scalar_coerce(ds.measure, val)


@dispatch(object, object)
def scalar_coerce(dtype, val):
    return val


@dispatch(_strtypes, object)
def scalar_coerce(ds, val):
    return scalar_coerce(dshape(ds), val)


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


class Relational(Arithmetic):
    _dtype = ct.bool_


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
    _dtype = ct.bool_


class Or(Arithmetic):
    symbol = '|'
    op = operator.or_
    _dtype = ct.bool_


class Not(UnaryOp):
    symbol = '~'
    op = operator.invert
    _dtype = ct.bool_
    def __str__(self):
        return '~%s' % parenthesize(eval_str(self._child))


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


from .expressions import schema_method_list

schema_method_list.extend([
    (isscalar,
            set([_add, _radd, _mul,
            _rmul, _div, _rdiv, _floordiv, _rfloordiv, _sub, _rsub, _pow,
            _rpow, _mod, _rmod,  _neg])),
    (isscalar, set([_eq, _ne, _lt, _le, _gt, _ge])),
    (isscalar, set([_or, _ror, _and, _rand, _invert])),
    ])
