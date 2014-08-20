from __future__ import absolute_import, division, print_function

import numpy as np
import tables as tb
from blaze.expr import Selection, Head, Column, ColumnWise, Projection
from blaze.expr import eval_str, Expr, TableSymbol, Sort, TrueDiv, FloorDiv
from datashape import Record
from ..dispatch import dispatch


@dispatch(tb.Table)
def discover(t):
    return t.shape[0] * Record([[col, t.coltypes[col]] for col in t.colnames])


@dispatch(TableSymbol, tb.Table)
def compute_one(sym, t, **kwargs):
    return t.read()


@dispatch(Selection, tb.Table)
def compute_one(sel, t, **kwargs):
    s = eval_str(sel.predicate.expr)
    return t.read_where(s)


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
    return t.col(c.column)


@dispatch(Head, tb.Table)
def compute_one(h, t, **kwargs):
    return t[:h.n]


@dispatch(FloorDiv, list, tb.Table)
def compute_one(expr, c, t, **kwargs):
    raise NotImplementedError('expr %r using // not implemented by numexpr' %
                              expr)


@dispatch(Expr, list, tb.Table)
def compute_one(expr, columns, t, **kwargs):
    uservars = dict((col, getattr(t.cols, col)) for col in columns)
    e = tb.Expr(str(expr), uservars=uservars, truediv=isinstance(expr, TrueDiv))
    return e.eval()


@dispatch(ColumnWise, tb.Table)
def compute_one(c, t, **kwargs):
    return compute_one(c.expr, c.active_columns(), t, **kwargs)


@dispatch(Sort, tb.Table)
def compute_one(s, t, **kwargs):
    result = np.sort(t[:], order=s.key)
    if s.ascending:
        return result
    return result[::-1]
