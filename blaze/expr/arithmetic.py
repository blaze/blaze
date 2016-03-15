from __future__ import absolute_import, division, print_function

import operator
from toolz import first
import numpy as np
import pandas as pd
from datashape import (
    DataShape,
    DateTime,
    Option,
    String,
    TimeDelta,
    coretypes as ct,
    datetime_,
    discover,
    dshape,
    optionify,
    promote,
    timedelta_,
    unsigned,
)
from datashape.predicates import isscalar, isboolean, isnumeric, isdatelike
from datashape.typesets import integral
from dateutil.parser import parse as dt_parse


from .core import parenthesize, eval_str
from .expressions import Expr, shape, ElemWise, binop_inputs, binop_name
from .utils import maxshape
from ..dispatch import dispatch
from ..compatibility import _strtypes


__all__ = '''
BinOp
UnaryOp
Arithmetic
Add
Mult
Repeat
Sub
Div
FloorDiv
Pow
Mod
Interp
USub
Relational
Eq
Ne
Ge
Lt
Le
Gt
Gt
And
Or
Not
'''.split()


class BinOp(ElemWise):
    __slots__ = '_hash', 'lhs', 'rhs'
    __inputs__ = 'lhs', 'rhs'

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
        self._hash = None

    def __str__(self):
        lhs = parenthesize(eval_str(self.lhs))
        rhs = parenthesize(eval_str(self.rhs))
        return '%s %s %s' % (lhs, self.symbol, rhs)

    def _dshape(self):
        # TODO: better inference.  e.g. int + int -> int
        return DataShape(*(maxshape([shape(self.lhs), shape(self.rhs)]) +
                           (self._dtype,)))

    _name = property(binop_name)


    @property
    def _inputs(self):
        return tuple(binop_inputs(self))


class UnaryOp(ElemWise):
    __slots__ = '_hash', '_child',

    def __init__(self, child):
        self._child = child
        self._hash = None

    def __str__(self):
        return '%s(%s)' % (self.symbol, eval_str(self._child))

    @property
    def symbol(self):
        return type(self).__name__

    def _dshape(self):
        return DataShape(*(shape(self._child) + (self._dtype,)))

    @property
    def _name(self):
        return self._child._name


class Arithmetic(BinOp):
    """ Super class for arithmetic operators like add or mul """

    @property
    def _dtype(self):
        # we can't simply use .schema or .datashape because we may have a bare
        # integer, for example
        lhs, rhs = discover(self.lhs).measure, discover(self.rhs).measure
        return promote(lhs, rhs)


class Add(Arithmetic):
    symbol = '+'
    op = operator.add

    @property
    def _dtype(self):
        lmeasure = discover(self.lhs).measure
        lty = getattr(lmeasure, 'ty', lmeasure)
        rmeasure = discover(self.rhs).measure
        rty = getattr(rmeasure, 'ty', rmeasure)
        if lmeasure == datetime_ and rmeasure == datetime_:
            raise TypeError('cannot add datetime to datetime')

        l_is_datetime = lty == datetime_
        if l_is_datetime or rty == datetime_:
            if l_is_datetime:
                other = rty
            else:
                other = lty
            if isinstance(other, TimeDelta):
                return optionify(lmeasure, rmeasure, datetime_)
            else:
                raise TypeError(
                    'can only add timedeltas to datetimes',
                )

        return super(Add, self)._dtype


class Mult(Arithmetic):
    symbol = '*'
    op = operator.mul


class Repeat(Arithmetic):
    # Sequence repeat
    symbol = '*'
    op = operator.mul

    @property
    def _dtype(self):
        lmeasure = discover(self.lhs).measure
        rmeasure = discover(self.rhs).measure
        if not (isinstance(getattr(lmeasure, 'ty', lmeasure), String) and
                getattr(rmeasure, 'ty', rmeasure) in integral):
            raise TypeError(
                'can only repeat strings by an integer amount, got: %s * %s' %
                (lmeasure, rmeasure),
            )

        return optionify(lmeasure, rmeasure, lmeasure)


class Sub(Arithmetic):
    symbol = '-'
    op = operator.sub

    @property
    def _dtype(self):
        lmeasure = discover(self.lhs).measure
        lty = getattr(lmeasure, 'ty', lmeasure)
        rmeasure = discover(self.rhs).measure
        rty = getattr(rmeasure, 'ty', rmeasure)
        if lty == datetime_:
            if isinstance(rty, DateTime):
                return optionify(lmeasure, rmeasure, timedelta_)
            if isinstance(rty, TimeDelta):
                return optionify(lmeasure, rmeasure, datetime_)
            else:
                raise TypeError(
                    'can only subtract timedelta or datetime from datetime',
                )

        return super(Sub, self)._dtype


class Div(Arithmetic):
    symbol = '/'
    op = operator.truediv

    @property
    def _dtype(self):
        lhs, rhs = discover(self.lhs).measure, discover(self.rhs).measure
        return optionify(lhs, rhs, ct.float64)


class FloorDiv(Arithmetic):
    symbol = '//'
    op = operator.floordiv

    @property
    def _dtype(self):
        lhs, rhs = discover(self.lhs).measure, discover(self.rhs).measure
        is_unsigned = lhs in unsigned and rhs in unsigned
        max_width = max(lhs.itemsize, rhs.itemsize)
        prefix = 'u' if is_unsigned else ''
        measure = getattr(ct, '%sint%d' % (prefix, max_width * 8))
        return optionify(lhs, rhs, measure)


class Pow(Arithmetic):
    symbol = '**'
    op = operator.pow


class Mod(Arithmetic):
    symbol = '%'
    op = operator.mod


class Interp(Arithmetic):
    # String interpolation
    symbol = '%'
    op = operator.mod

    @property
    def _dtype(self):
        lmeasure = discover(self.lhs).measure
        rmeasure = discover(self.rhs).measure
        if not (isinstance(getattr(lmeasure, 'ty', lmeasure), String)):
            raise TypeError('can only interp strings got: %s' % lmeasure)

        return optionify(lmeasure, rmeasure, lmeasure)


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
    return scalar_coerce(ds.ty, val) if val is not None else None


@dispatch((ct.Record, ct.Mono, ct.Option, DataShape), Expr)
def scalar_coerce(ds, val):
    return val


@dispatch(ct.Date, _strtypes)
def scalar_coerce(_, val):
    if val == '':
        raise TypeError('%r is not a valid date' % val)
    dt = dt_parse(val)
    if dt.time():  # TODO: doesn't work with python 3.5
        raise TypeError(
            "Can not coerce %r to type Date, contains time information" % val
        )
    return dt.date()


@dispatch(ct.DateTime, _strtypes)
def scalar_coerce(_, val):
    if val == '':
        raise TypeError('%r is not a valid datetime' % val)
    return pd.Timestamp(val)


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


def _mkbin(name, cons, private=True, reflected=True):
    prefix = '_' if private else ''

    def _bin(self, other):
        result = cons(self, scalar_coerce(self.dshape, other))
        result.dshape  # Check that shapes and dtypes match up
        return result
    _bin.__name__ = prefix + name

    if reflected:
        def _rbin(self, other):
            result = cons(scalar_coerce(self.dshape, other), self)
            result.dshape  # Check that shapes and dtypes match up
            return result
        _rbin.__name__ = prefix + 'r' + name

        return _bin, _rbin

    return _bin


_add, _radd = _mkbin('add', Add)
_div, _rdiv = _mkbin('div', Div)
_floordiv, _rfloordiv = _mkbin('floordiv', FloorDiv)
_mod, _rmod = _mkbin('mod', Mod)
_mul, _rmul = _mkbin('mul', Mult)
_pow, _rpow = _mkbin('pow', Pow)
repeat = _mkbin('repeat', Repeat, reflected=False, private=False)
_sub, _rsub = _mkbin('sub', Sub)
interp = _mkbin('interp', Interp, reflected=False, private=False)


class _Optional(Arithmetic):
    @property
    def _dtype(self):
        # we can't simply use .schema or .datashape because we may have a bare
        # integer, for example
        lhs, rhs = discover(self.lhs).measure, discover(self.rhs).measure
        if isinstance(lhs, Option) or isinstance(rhs, Option):
            return Option(ct.bool_)
        return ct.bool_


class Relational(_Optional):
    # Leave this to separate relationals from other types of optionals.
    pass


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


class And(_Optional):
    symbol = '&'
    op = operator.and_


class Or(_Optional):
    symbol = '|'
    op = operator.or_


class Not(UnaryOp):
    symbol = '~'
    op = operator.invert

    @property
    def _dtype(self):
        return self._child.schema

    def __str__(self):
        return '~%s' % parenthesize(eval_str(self._child))


_and, _rand = _mkbin('and', And)
_eq = _mkbin('eq', Eq, reflected=False)
_ge = _mkbin('ge', Ge, reflected=False)
_gt = _mkbin('gt', Gt, reflected=False)
_le = _mkbin('le', Le, reflected=False)
_lt = _mkbin('lt', Lt, reflected=False)
_ne = _mkbin('ne', Ne, reflected=False)
_or, _ror = _mkbin('or', Or)


def _invert(self):
    result = Invert(self)
    result.dshape  # Check that shapes and dtypes match up
    return result


Invert = Not
BitAnd = And
BitOr = Or


from .expressions import schema_method_list


schema_method_list.extend([
    (isnumeric,
     set([_add, _radd, _mul, _rmul, _div, _rdiv, _floordiv, _rfloordiv, _sub,
          _rsub, _pow, _rpow, _mod, _rmod,  _neg])),
    (isscalar, set([_eq, _ne, _lt, _le, _gt, _ge])),
    (isboolean, set([_or, _ror, _and, _rand, _invert])),
    (isdatelike, set([_add, _radd, _sub, _rsub])),
    ])
