from __future__ import absolute_import, division, print_function

import tables as tb
from blaze.expr.table import *
from datashape import Record
from ..dispatch import dispatch


@dispatch(tb.Table)
def discover(t):
    return t.shape[0] * Record([[col, t.coltypes[col]] for col in t.colnames])


@dispatch(Selection, tb.Table)
def compute_one(sel, t, **kwargs):
    s = eval_str(sel.predicate.expr)
    return t.read_where(s)


@dispatch(Head, tb.Table)
def compute_one(h, t, **kwargs):
    return t[:h.n]
