from __future__ import absolute_import, print_function, division

from graphlab import SFrame, SArray
import graphlab.aggregate as agg
from itertools import chain

import pandas as pd
from odo import convert
from toolz import unique, concat

from .core import base
from datashape import Record, string, int64, float64
from blaze import discover, dispatch, compute
from blaze.compute.core import compute_up
from blaze.expr import (Projection, Field, Reduction, Head, Expr, BinOp, Sort,
                        By, Join, Selection)


python_type_to_datashape = {
    str: string,
    int: int64,
    float: float64,
}


@discover.register(SFrame)
def discover_sframe(sf):
    types = [python_type_to_datashape[t] for t in sf.dtype()]
    return len(sf) * Record(list(zip(sf.column_names(), types)))


@dispatch(Projection, SFrame)
def compute_up(expr, data, **kwargs):
    return data[expr.columns]


@dispatch(Field, SFrame)
def compute_up(expr, data, **kwargs):
    return data[expr._name]


@dispatch(Reduction, SArray)
def compute_up(expr, data, **kwargs):
    return getattr(data, expr.symbol)()


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


@dispatch(By, SFrame)
def compute_up(expr, data, **kwargs):
    app = expr.apply
    operations = dict((k, getattr(agg, v.symbol.upper())(v._child._name))
                      for k, v in zip(app.fields, app.values))
    return data.groupby(key_columns=expr.grouper.fields,
                        operations=operations)


@dispatch(Join, SFrame, SFrame)
def compute_up(expr, lhs, rhs, **kwargs):
    # TODO: make sure this is robust to the kind of join
    columns = list(unique(chain(lhs.column_names(), rhs.column_names())))
    on = list(concat(unique((tuple(expr.on_left), tuple(expr.on_right)))))
    return lhs.join(rhs, on=on, how=expr.how)[columns]


@convert.register(pd.DataFrame, SFrame)
def convert_sframe_to_dataframe(sf, **kwargs):
    return sf.to_dataframe()


@convert.register(pd.Series, SArray)
def convert_sarray_to_series(sa, **kwargs):
    return pd.Series(sa)
