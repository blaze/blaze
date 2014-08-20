from __future__ import absolute_import, division, print_function

import numpy as np
import tables as tb
from blaze.expr import Selection, Head, Column, ColumnWise, Projection
from blaze.expr import eval_str, Sort
from datashape import Record
from ..dispatch import dispatch


@dispatch(tb.Table)
def discover(t):
    return t.shape[0] * Record([[col, t.coltypes[col]] for col in t.colnames])


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


@dispatch(ColumnWise, tb.Table)
def compute_one(c, t, **kwargs):
    uservars = dict((col, getattr(t.cols, col)) for col in c.active_columns())
    e = tb.Expr(str(c.expr), uservars=uservars, truediv=True)
    return e.eval()


@dispatch(Sort, tb.Table)
def compute_one(s, t, **kwargs):
    result = t.read_sorted(sortby=s.key, checkCSI=True)
    if s.ascending:
        return result
    return result[::-1]
