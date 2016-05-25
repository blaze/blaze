from __future__ import absolute_import, division, print_function

from functools import partial
import numpy as np
import tables as tb
from datashape import Record, from_numpy, datetime_, date_

from blaze.expr import (Selection, Head, Field, Broadcast, Projection, Symbol,
                        Sort, Reduction, count, Slice, Expr, nelements,
                        UnaryOp, BinOp)
from blaze.compatibility import basestring, map
from blaze.expr.optimize import simple_selections
from ..dispatch import dispatch
from .numexpr import broadcast_numexpr_collect, print_numexpr


__all__ = ['drop', 'create_index']


@dispatch(tb.Table)
def discover(t):
    return t.shape[0] * Record([[col, discover(getattr(t.cols, col))]
                                for col in t.colnames])


@dispatch(tb.Column)
def discover(c):
    dshape = from_numpy(c.shape, c.dtype)
    return {'time64': datetime_, 'time32': date_}.get(c.type,
                                                      dshape.subshape[1])


@dispatch(tb.Table)
def drop(t):
    t.remove()


@dispatch(tb.Table, basestring)
def create_index(t, column, name=None, **kwargs):
    create_index(getattr(t.cols, column), **kwargs)


@dispatch(tb.Table, (list, tuple))
def create_index(t, columns, name=None, **kwargs):
    if not all(map(partial(hasattr, t.cols), columns)):
        raise ValueError('table %s does not have all passed in columns %s' %
                         (t, columns))
    for column in columns:
        create_index(t, column, **kwargs)


@dispatch(tb.Column)
def create_index(c, optlevel=9, kind='full', name=None, **kwargs):
    c.create_index(optlevel=optlevel, kind=kind, **kwargs)


@dispatch(Selection, tb.Table)
def compute_up(expr, data, **kwargs):
    predicate = optimize(expr.predicate, data)
    assert isinstance(predicate, Broadcast)

    s = predicate._scalars[0]
    cols = [s[field] for field in s.fields]
    expr_str = print_numexpr(cols, predicate._scalar_expr)
    return data.read_where(expr_str)


@dispatch(Symbol, tb.Table)
def compute_up(ts, t, **kwargs):
    return t


@dispatch(Reduction, (tb.Column, tb.Table))
def compute_up(r, c, **kwargs):
    return compute_up(r, c[:])


@dispatch(Projection, tb.Table)
def compute_up(proj, t, **kwargs):
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
    columns = proj.fields
    dtype = np.dtype([(col, t.dtype[col]) for col in columns])
    out = np.empty(t.shape, dtype=dtype)
    for c in columns:
        out[c] = t.col(c)
    return out


@dispatch(Field, tb.File)
def compute_up(expr, data, **kwargs):
    return data.getNode('/')._v_children[expr._name]


@dispatch(Field, tb.Node)
def compute_up(expr, data, **kwargs):
    return data._v_children[expr._name]


@dispatch(Field, tb.Table)
def compute_up(c, t, **kwargs):
    return getattr(t.cols, c._name)


@dispatch(count, tb.Column)
def compute_up(r, c, **kwargs):
    return len(c)


@dispatch(Head, (tb.Column, tb.Table))
def compute_up(h, t, **kwargs):
    return t[:h.n]


@dispatch(Broadcast, tb.Table)
def compute_up(expr, data, **kwargs):
    if len(expr._children) != 1:
        raise ValueError("Only one child in Broadcast allowed")

    s = expr._scalars[0]
    cols = [s[field] for field in s.fields]
    expr_str = print_numexpr(cols, expr._scalar_expr)
    uservars = dict((c, getattr(data.cols, c)) for c in s.fields)
    e = tb.Expr(expr_str, uservars=uservars, truediv=True)
    return e.eval()


@dispatch(Sort, tb.Table)
def compute_up(s, t, **kwargs):
    if isinstance(s.key, Field) and s.key._child.isidentical(s._child):
        key = s.key._name
    else:
        key = s.key
    assert hasattr(t.cols, key), 'Table has no column(s) %s' % s.key
    result = t.read_sorted(sortby=key, checkCSI=True)
    if s.ascending:
        return result
    return result[::-1]


@dispatch(Slice, (tb.Table, tb.Column))
def compute_up(expr, x, **kwargs):
    return x[expr.index]


Broadcastable = UnaryOp, BinOp, Field
WantToBroadcast = UnaryOp, BinOp


@dispatch(Expr, tb.Table)
def optimize(expr, seq):
    return broadcast_numexpr_collect(
        simple_selections(expr),
        broadcastable=Broadcastable,
        want_to_broadcast=WantToBroadcast,
        no_recurse=Selection,
    )


@dispatch(nelements, tb.Table)
def compute_up(expr, x, **kwargs):
    return compute_up.dispatch(type(expr), np.ndarray)(expr, x, **kwargs)
