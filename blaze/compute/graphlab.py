from __future__ import absolute_import, print_function, division

from graphlab import SFrame, SArray
import graphlab.aggregate as agg
from itertools import chain

import array
import pandas as pd
from odo import convert
from toolz import unique, concat
from cytoolz import take

from .core import base
from datashape import Record, string, int64, float64, Option, var
from blaze import discover, dispatch, compute
from blaze.compute.core import compute_up
from blaze.expr import (Projection, Field, Reduction, Head, Expr, BinOp, Sort,
                        By, Join, Selection, common_subexpression, nelements)


python_type_to_datashape = {
    str: string,
    int: int64,
    float: float64,
}


@discover.register(SFrame)
def discover_sframe(sf, n=1000):
    columns = sf.column_names()
    types = map(lambda x, n=n: discover(x, n=n).measure,
                (sf[name] for name in columns))
    return var * Record(list(zip(columns, types)))


@discover.register(SArray)
def discover_sarray(sa, n=1000):
    dtype = sa.dtype()
    if issubclass(dtype, (dict, list, array.array)):
        return var * discover(list(take(n, sa))).measure
    return var * Option(python_type_to_datashape[dtype])


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
    columns = list(unique(chain(lhs.column_names(), rhs.column_names())))
    on = list(concat(unique((tuple(expr.on_left), tuple(expr.on_right)))))
    return lhs.join(rhs, on=on, how=expr.how)[columns]


@convert.register(pd.DataFrame, SFrame)
def convert_sframe_to_dataframe(sf, **kwargs):
    return sf.to_dataframe()


@convert.register(pd.Series, SArray)
def convert_sarray_to_series(sa, **kwargs):
    return pd.Series(sa)
