from __future__ import absolute_import, print_function, division

from graphlab import SFrame, SArray
import graphlab.aggregate as agg
from itertools import chain

from toolz import unique, concat

from .core import base
from blaze import dispatch, compute
from blaze.compute.core import compute_up
from blaze.expr import (Projection, Field, Reduction, Head, Expr, BinOp, Sort,
                        By, Join, Selection, common_subexpression, nelements)


@dispatch(Projection, SFrame)
def compute_up(expr, data, **kwargs):
    return data[expr.columns]


@dispatch(Field, SFrame)
def compute_up(expr, data, **kwargs):
    return data[expr._name]


@dispatch(Field, SArray)
def compute_up(expr, data, **kwargs):
    return data.unpack('', limit=expr.fields)[expr._name]


@dispatch(Reduction, SArray)
def compute_up(expr, data, **kwargs):
    return getattr(data, expr.symbol)()


@dispatch(nelements, SFrame)
def compute_up(expr, data, **kwargs):
    if expr.axis != (0,):
        raise ValueError('axis != 0 not allowed on tables')
    return data.num_rows()


@dispatch(nelements, SArray)
def compute_up(expr, data, **kwargs):
    if expr.axis != (0,):
        raise ValueError('axis != 0 not allowed on vectors')
    return data.size()


@dispatch(Sort, SArray)
def compute_up(expr, data, **kwargs):
    return data.sort(ascending=expr.ascending)


@dispatch(Sort, SFrame)
def compute_up(expr, data, **kwargs):
    return data.sort(list(expr._key), ascending=expr.ascending)


@dispatch(Head, (SFrame, SArray))
def compute_up(expr, data, **kwargs):
    return data.head(expr.n)


@dispatch(BinOp, SArray)
def compute_up(expr, data, **kwargs):
    if isinstance(expr.lhs, Expr):
        return expr.op(data, expr.rhs)
    else:
        return expr.op(expr.lhs, data)


@compute_up.register(BinOp, SArray, (SArray, base))
@compute_up.register(BinOp, base, SArray)
def compute_up_binop_sarray_base(expr, lhs, rhs, **kwargs):
    return expr.op(lhs, rhs)


@dispatch(Selection, (SFrame, SArray))
def compute_up(expr, data, **kwargs):
    return data[compute(expr.predicate, {expr._child: data})]


@dispatch(By, (SFrame, SArray))
def compute_up(expr, data, **kwargs):
    app = expr.apply
    operations = dict((k, getattr(agg, v.symbol.upper())(v._child._name))
                      for k, v in zip(app.fields, app.values))

    # get the fields (meaning we're a dict dtype) if we're an SArray
    if isinstance(data, SArray):
        all_fields = frozenset(common_subexpression(expr.grouper, app).fields)

        # only rip out the fields we need
        limit = list(chain(expr.grouper.fields,
                           (f._name for f in app._subterms()
                            if isinstance(f, Field) and
                               f._name in all_fields)))
        data = data.unpack('', limit=limit)
    return data.groupby(expr.grouper.fields, operations=operations)


@dispatch(Join, SFrame, SFrame)
def compute_up(expr, lhs, rhs, **kwargs):
    # TODO: make sure this is robust to the kind of join
    columns = list(unique(chain(expr.lhs.fields, expr.rhs.fields)))
    on = list(concat(unique((tuple(expr.on_left), tuple(expr.on_right)))))
    return lhs.join(rhs, on=on, how=expr.how)[columns]
