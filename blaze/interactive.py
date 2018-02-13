from __future__ import absolute_import, division, print_function

from functools import reduce
import operator
import warnings

import datashape
from datashape import var, Map
from datashape.predicates import (
    isscalar,
    iscollection,
    isrecord,
    istabular,
)
from odo import odo
from pandas import DataFrame, Series

import blaze
from .compute import compute
from .compute.core import coerce_scalar
from .expr import Expr, Symbol, ndim
from .dispatch import dispatch
from .compatibility import _strtypes


__all__ = ['to_html']


def data(*args, **kwargs):
    warnings.warn(DeprecationWarning(
        'blaze.interactive.data has been moved to blaze.data'))
    return blaze.expr.literal.data(*args, **kwargs)


def concrete_head(expr, n=10):
    """ Return head of computed expression """
    if not expr._resources():
        raise ValueError("Expression does not contain data resources")
    if not iscollection(expr.dshape):
        return compute(expr)

    head = expr.head(n + 1)

    if not iscollection(expr.dshape):
        return odo(head, object)
    elif isrecord(expr.dshape.measure):
        return odo(head, DataFrame)

    df = odo(head, DataFrame)
    df.columns = [expr._name]
    return df


def _peek_tables(expr, n=10):
    return concrete_head(expr, n).rename(columns={None: ''})


def repr_tables(expr, n=10):
    result = concrete_head(expr, n).rename(columns={None: ''})

    if isinstance(result, (DataFrame, Series)):
        s = repr(result)
        if len(result) > 10:
            s = '\n'.join(s.split('\n')[:-1]) + '\n...'
        return s
    else:
        return result.peek()  # pragma: no cover


def numel(shape):
    if var in shape:
        return None
    if not shape:
        return 1
    return reduce(operator.mul, shape, 1)


def short_dshape(ds, nlines=5):
    s = datashape.coretypes.pprint(ds)
    lines = s.split('\n')
    if len(lines) > 5:
        s = '\n'.join(lines[:nlines]) + '\n  ...'
    return s


def _peek(expr):
    # Pure Expressions, not interactive
    if not set(expr._resources().keys()).issuperset(expr._leaves()):
        return expr

    # Scalars
    if ndim(expr) == 0 and isscalar(expr.dshape):
        return coerce_scalar(compute(expr), str(expr.dshape))

    # Tables
    if (ndim(expr) == 1 and (istabular(expr.dshape) or
                             isscalar(expr.dshape.measure) or
                             isinstance(expr.dshape.measure, Map))):
        return _peek_tables(expr, 10)

    # Smallish arrays
    if ndim(expr) >= 2 and numel(expr.shape) and numel(expr.shape) < 1000000:
        return compute(expr)

    # Other
    dat = expr._resources().values()
    if len(dat) == 1:
        dat = list(dat)[0]  # may be dict_values
    return dat


def expr_repr(expr, n=10):
    # Pure Expressions, not interactive
    if not set(expr._resources().keys()).issuperset(expr._leaves()):
        return str(expr)

    # Scalars
    if ndim(expr) == 0 and isscalar(expr.dshape):
        return repr(coerce_scalar(compute(expr), str(expr.dshape)))

    # Tables
    if (ndim(expr) == 1 and (istabular(expr.dshape) or
                             isscalar(expr.dshape.measure) or
                             isinstance(expr.dshape.measure, Map))):
        return repr_tables(expr, 10)

    # Smallish arrays
    if ndim(expr) >= 2 and numel(expr.shape) and numel(expr.shape) < 1000000:
        return repr(compute(expr))

    # Other
    dat = expr._resources().values()
    if len(dat) == 1:
        dat = list(dat)[0]  # may be dict_values

    s = 'Data:       %s' % dat
    if not isinstance(expr, Symbol):
        s += '\nExpr:       %s' % str(expr)
    s += '\nDataShape:  %s' % short_dshape(expr.dshape, nlines=7)

    return s


@dispatch(DataFrame)
def to_html(df):
    return df.to_html()


@dispatch(Expr)
def to_html(expr):
    # Tables
    if not expr._resources() or ndim(expr) != 1:
        return to_html(expr_repr(expr))
    return to_html(concrete_head(expr))


@dispatch(object)
def to_html(o):
    return repr(o)


@dispatch(_strtypes)
def to_html(o):
    return o.replace('\n', '<br>')


def table_length(expr):
    try:
        return expr._len()
    except ValueError:
        return int(expr.count())


Expr.peek = _peek
Expr.__len__ = table_length


def convert_base(typ, x):
    x = compute(x)
    try:
        return typ(x)
    except:
        return typ(odo(x, typ))


Expr.__int__ = lambda x: convert_base(int, x)
Expr.__float__ = lambda x: convert_base(float, x)
Expr.__complex__ = lambda x: convert_base(complex, x)
Expr.__bool__ = lambda x: convert_base(bool, x)
Expr.__nonzero__ = lambda x: convert_base(bool, x)
