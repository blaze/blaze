from __future__ import absolute_import, division, print_function

from functools import partial
import numpy as np
import tables as tb
from blaze.expr import (Selection, Head, Column, ColumnWise, Projection,
                        TableSymbol, Sort, Reduction, count)
from blaze.expr import eval_str
from blaze.compatibility import basestring, map
from datashape import Record
from ..dispatch import dispatch


@dispatch(tb.Table)
def discover(t):
    return t.shape[0] * Record([[col, t.coltypes[col]] for col in t.colnames])


@dispatch(tb.Table)
def drop(t):
    t.remove()


@dispatch(tb.Table, basestring)
def create_index(t, column, **kwargs):
    create_index(getattr(t.cols, column), **kwargs)


@dispatch(tb.Table, list)
def create_index(t, columns, **kwargs):
    if not all(map(partial(hasattr, t.cols), columns)):
        raise ValueError('table %s does not have all passed in columns %s' %
                         (t, columns))
    for column in columns:
        create_index(t, column, **kwargs)


@dispatch(tb.Column)
def create_index(c, optlevel=9, kind='full', **kwargs):
    c.create_index(optlevel=optlevel, kind=kind, **kwargs)


@dispatch(Selection, tb.Table)
def compute_one(sel, t, **kwargs):
    s = eval_str(sel.predicate.expr)
    return t.read_where(s)


@dispatch(TableSymbol, tb.Table)
def compute_one(ts, t, **kwargs):
    return t


@dispatch(Reduction, (tb.Column, tb.Table))
def compute_one(r, c, **kwargs):
    return compute_one(r, c[:])


@dispatch(Projection, tb.Table)
def compute_one(proj, t, **kwargs):
    # only options here are
    # read the whole thing in and then select
    # or
    # create an output array that is at most the size of the on disk table and
    # fill it will the columns iteratively
    # both of these options aren't ideal but pytables has no way to select
    # multiple column subsets so pick the one where we can optimize for the best
    # case rather than prematurely pessimizing
    #
    # TODO: benchmark on big tables because i'm not sure exactly what the
    # implications here are for memory usage
    columns = proj.columns
    dtype = np.dtype([(col, t.dtype[col]) for col in columns])
    out = np.empty(t.shape, dtype=dtype)
    for c in columns:
        out[c] = t.col(c)
    return out


@dispatch(Column, tb.Table)
def compute_one(c, t, **kwargs):
    return getattr(t.cols, c.column)


@dispatch(count, tb.Column)
def compute_one(r, c, **kwargs):
    return len(c)


@dispatch(Head, tb.Table)
def compute_one(h, t, **kwargs):
    return t[:h.n]


@dispatch(ColumnWise, tb.Table)
def compute_one(c, t, **kwargs):
    uservars = dict((col, getattr(t.cols, col)) for col in c.active_columns())
    e = tb.Expr(str(c.expr), uservars=uservars, truediv=True)
    return e.eval()


@dispatch(Sort, tb.Table)
def compute_one(s, t, **kwargs):
    if isinstance(s.key, Column) and s.key.child.isidentical(s.child):
        key = s.key.name
    else:
        key = s.key
    assert hasattr(t.cols, key), 'Table has no column(s) %s' % s.key
    result = t.read_sorted(sortby=key, checkCSI=True)
    if s.ascending:
        return result
    return result[::-1]
