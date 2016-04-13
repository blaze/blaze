"""

>>> from blaze.expr import symbol
>>> from blaze.compute.pandas import compute

>>> accounts = symbol('accounts', 'var * {name: string, amount: int}')
>>> deadbeats = accounts[accounts['amount'] < 0]['name']

>>> from pandas import DataFrame
>>> data = [['Alice', 100], ['Bob', -50], ['Charlie', -20]]
>>> df = DataFrame(data, columns=['name', 'amount'])
>>> compute(deadbeats, df)
1        Bob
2    Charlie
Name: name, dtype: object
"""
from __future__ import absolute_import, division, print_function

from datetime import timedelta
import fnmatch
import itertools
from distutils.version import LooseVersion
import warnings
from collections import defaultdict

import numpy as np

import pandas as pd

from pandas.core.generic import NDFrame
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy

from toolz import merge as merge_dicts
from toolz import groupby
from toolz.curried import pipe, filter, map, concat

import datashape

from datashape import to_numpy_dtype
from datashape.predicates import isscalar

from odo import into
try:
    import dask.dataframe as dd
    DaskDataFrame = dd.DataFrame
    DaskSeries = dd.Series
except ImportError:
    DaskDataFrame = pd.DataFrame
    DaskSeries = pd.Series

from .core import compute, compute_up, base
from ..compatibility import _inttypes
from ..dispatch import dispatch
from ..expr import (
    Apply,
    BinOp,
    Broadcast,
    By,
    Coalesce,
    Coerce,
    Concat,
    DateTime,
    DateTimeTruncate,
    Distinct,
    ElemWise,
    Expr,
    Field,
    Head,
    Interp,
    IsIn,
    Join,
    Label,
    Like,
    Map,
    Merge,
    Millisecond,
    Projection,
    ReLabel,
    Reduction,
    Sample,
    Selection,
    Shift,
    Slice,
    Sort,
    Summary,
    Tail,
    UTCFromTimestamp,
    UnaryOp,
    UnaryStringFunction,
    common_subexpression,
    count,
    isnan,
    nelements,
    notnull,
    nunique,
    std,
    summary,
    symbol,
    var,
    StrCat,
)

__all__ = []


@dispatch(Projection, (DataFrame, DaskDataFrame))
def compute_up(t, df, **kwargs):
    return df[list(t.fields)]


@dispatch(Field, (DataFrame, DataFrameGroupBy, DaskDataFrame))
def compute_up(t, df, **kwargs):
    assert len(t.fields) == 1
    return df[t.fields[0]]


@dispatch(Field, (Series, DaskSeries))
def compute_up(t, data, **kwargs):
    assert len(t.fields) == 1
    if t.fields[0] == data.name:
        return data
    else:
        raise ValueError("Fieldname %r does not match Series name %r"
                         % (t.fields[0], data.name))


@dispatch(Broadcast, (DataFrame, DaskDataFrame))
def compute_up(t, df, **kwargs):
    return compute(t._full_expr, df, return_type='native')


@dispatch(Broadcast, Series)
def compute_up(t, s, **kwargs):
    return compute_up(t, s.to_frame(), **kwargs)


@dispatch(Interp, Series)
def compute_up(t, data, **kwargs):
    if isinstance(t.lhs, Expr):
        return data % t.rhs
    else:
        return t.lhs % data


@compute_up.register(Interp, Series, (Series, base))
@compute_up.register(Interp, base, Series)
def compute_up_pd_interp(t, lhs, rhs, **kwargs):
    return lhs % rhs



@dispatch(BinOp, (Series, DaskSeries))
def compute_up(t, data, **kwargs):
    if isinstance(t.lhs, Expr):
        return t.op(data, t.rhs)
    else:
        return t.op(t.lhs, data)

@compute_up.register(BinOp, (Series, DaskSeries), (Series, base, DaskSeries))
@compute_up.register(BinOp, (Series, base, DaskSeries), (Series, DaskSeries))
def compute_up_binop(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(UnaryOp, NDFrame)
def compute_up(t, df, **kwargs):
    f = getattr(t, 'op', getattr(np, t.symbol, None))
    if f is None:
        raise ValueError('%s is not a valid operation on %s objects' %
                         (t.symbol, type(df).__name__))
    return f(df)


@dispatch(Selection, (Series, DataFrame, DaskSeries, DaskDataFrame))
def compute_up(expr, df, **kwargs):
    return compute_up(
        expr,
        df,
        compute(expr.predicate, {expr._child: df}, return_type='native'),
        **kwargs
    )


@dispatch(Selection, (Series, DataFrame, DaskSeries, DaskDataFrame),
                     (Series, DaskSeries))
def compute_up(expr, df, predicate, **kwargs):
    return df[predicate]


@dispatch(Join, DataFrame, DataFrame)
def compute_up(t, lhs, rhs, **kwargs):
    """ Join two pandas data frames on arbitrary columns

    The approach taken here could probably be improved.

    To join on two columns we force each column to be the index of the
    dataframe, perform the join, and then reset the index back to the left
    side's original index.
    """
    result = pd.merge(
        lhs,
        rhs,
        left_on=t.on_left,
        right_on=t.on_right,
        how=t.how,
        suffixes=t.suffixes,
    )
    return result.reset_index()[t.fields]


@dispatch(isnan, pd.Series)
def compute_up(expr, data, **kwargs):
    return data.isnull()


@dispatch(notnull, pd.Series)
def compute_up(expr, data, **kwargs):
    return data.notnull()


pandas_structure = DataFrame, DaskDataFrame, Series, DataFrameGroupBy, SeriesGroupBy


@dispatch(Concat, pandas_structure, pandas_structure)
def compute_up(t, lhs, rhs, _concat=pd.concat, **kwargs):
    if not (isinstance(lhs, type(rhs)) or isinstance(rhs, type(lhs))):
        raise TypeError('lhs and rhs must be the same type')

    return _concat((lhs, rhs), axis=t.axis, ignore_index=True)


def get_scalar(result):
    # pandas may return an int, numpy scalar or non scalar here so we need to
    # program defensively so that things are JSON serializable
    try:
        return result.item()
    except (AttributeError, ValueError):
        return result


@dispatch(Reduction, (Series, SeriesGroupBy, DaskSeries))
def compute_up(t, s, **kwargs):
    result = get_scalar(getattr(s, t.symbol)())
    if t.keepdims:
        result = Series([result], name=s.name)
    return result


@dispatch((std, var), (Series, SeriesGroupBy))
def compute_up(t, s, **kwargs):
    measure = t.schema.measure
    is_timedelta = isinstance(
        getattr(measure, 'ty', measure),
        datashape.TimeDelta,
    )
    if is_timedelta:
        # part 1 of 2 to work around the fact that pandas does not have
        # timedelta var or std: cast to a double
        s = s.astype('timedelta64[s]').astype('int64')
    result = get_scalar(getattr(s, t.symbol)(ddof=t.unbiased))
    if t.keepdims:
        result = Series([result], name=s.name)
    if is_timedelta:
        # part 2 of 2 to work around the fact that postgres does not have
        # timedelta var or std: cast back from seconds by creating a timedelta
        result = timedelta(seconds=result)
    return result


@dispatch(Distinct, DataFrame)
def compute_up(t, df, **kwargs):
    return df.drop_duplicates(subset=t.on or None).reset_index(drop=True)


@dispatch(Distinct, Series)
def compute_up(t, s, **kwargs):
    if t.on:
        raise ValueError('malformed expression: no columns to distinct on')
    return s.drop_duplicates().reset_index(drop=True)


@dispatch(nunique, DataFrame)
def compute_up(expr, data, **kwargs):
    return compute_up(expr._child.distinct().count(), data, **kwargs)


string_func_names = {'str_len': 'len',
                     'strlen': 'len',
                     'str_upper': 'upper',
                     'str_lower': 'lower'}


@dispatch(UnaryStringFunction, Series)
def compute_up(expr, data, **kwargs):
    name = type(expr).__name__
    return getattr(data.str, string_func_names.get(name, name))()


@dispatch(StrCat, Series, Series)
def compute_up(expr, lhs_data, rhs_data, **kwargs):
    res = lhs_data.str.cat(rhs_data, sep=expr.sep)
    return res


def unpack(seq):
    """ Unpack sequence of length one

    >>> unpack([1, 2, 3])
    [1, 2, 3]

    >>> unpack([1])
    1
    """
    seq = list(seq)
    if len(seq) == 1:
        seq = seq[0]
    return seq


Grouper = ElemWise, Series, list


@dispatch(By, list, DataFrame)
def get_grouper(c, grouper, df):
    return grouper


@dispatch(By, Expr, NDFrame)
def get_grouper(c, grouper, df):
    g = compute(grouper, {c._child: df}, return_type='native')
    if isinstance(g, Series):
        return g
    if isinstance(g, DataFrame):
        return [g[col] for col in g.columns]


@dispatch(By, (Field, Projection), NDFrame)
def get_grouper(c, grouper, df):
    return grouper.fields


@dispatch(By, Reduction, Grouper, NDFrame)
def compute_by(t, r, g, df):
    names = [r._name]
    preapply = compute(r._child, {t._child: df}, return_type='native')

    # Pandas and Blaze column naming schemes differ
    # Coerce DataFrame column names to match Blaze's names
    preapply = preapply.copy()
    if isinstance(preapply, Series):
        preapply.name = names[0]
    else:
        preapply.names = names
    group_df = concat_nodup(df, preapply)

    gb = group_df.groupby(g)
    groups = gb[names[0] if isscalar(t.apply._child.dshape.measure) else names]

    return compute_up(r, groups, return_type='native')  # do reduction


name_dict = dict()
seen_names = set()


def _name(expr):
    """ A unique and deterministic name for an expression """
    if expr in name_dict:
        return name_dict[expr]
    result = base = expr._name or '_'
    if result in seen_names:
        for i in itertools.count(1):
            result = '%s_%d' % (base, i)
            if result not in seen_names:
                break
    # result is an unseen name
    seen_names.add(result)
    name_dict[expr] = result
    return result


def fancify_summary(expr):
    """ Separate a complex summary into two pieces

    Helps pandas compute_by on summaries

    >>> t = symbol('t', 'var * {x: int, y: int}')
    >>> one, two, three = fancify_summary(summary(a=t.x.sum(), b=t.x.sum() + t.y.count() - 1))

    A simpler summary with only raw reductions
    >>> one
    summary(x_sum=sum(t.x), y_count=count(t.y))

    A mapping of those names to new leaves to use in another compuation
    >>> two  # doctest: +SKIP
    {'x_sum': x_sum, 'y_count': y_count}

    A mapping of computations to do for each column
    >>> three   # doctest: +SKIP
    {'a': x_sum, 'b': (x_sum + y_count) - 1}

    In this way, ``compute_by`` is able to do simple pandas reductions using
    groups.agg(...) and then do columnwise arithmetic afterwards.
    """
    seen_names.clear()
    name_dict.clear()
    exprs = pipe(expr.values,
                 map(Expr._traverse),
                 concat,
                 filter(lambda x: isinstance(x, Reduction)),
                 set)
    one = summary(**dict((_name(expr), expr) for expr in exprs))

    two = dict((_name(expr), symbol(_name(expr), datashape.var * expr.dshape))
               for expr in exprs)

    d = dict((expr, two[_name(expr)]) for expr in exprs)
    three = dict((name, value._subs(d)) for name, value in zip(expr.names,
                                                               expr.values))

    return one, two, three


@dispatch(By, Summary, Grouper, NDFrame)
def compute_by(t, s, g, df):
    one, two, three = fancify_summary(s)  # see above

    names_columns = list(zip(one.fields, one.values))
    func = lambda x: not isinstance(x[1], count)
    is_field = defaultdict(lambda: iter([]), groupby(func, names_columns))

    preapply = DataFrame(dict(
        zip([name for name, _ in is_field[True]],
            [compute(col._child, {t._child: df}, return_type='native')
             for _, col in is_field[True]])
        )
    )

    if list(is_field[False]):
        emptys = DataFrame([0] * len(df.index),
                           index=df.index,
                           columns=[name for name, _ in is_field[False]])
        preapply = concat_nodup(preapply, emptys)
    if not df.index.equals(preapply.index):
        df = df.loc[preapply.index]
    df2 = concat_nodup(df, preapply)

    groups = df2.groupby(g)

    d = dict((name, v.symbol) for name, v in zip(one.names, one.values))

    result = groups.agg(d)

    scope = dict((v, result[k]) for k, v in two.items())
    cols = [
        compute(expr.label(name), scope, return_type='native')
        for name, expr in three.items()
    ]

    result2 = pd.concat(cols, axis=1)

    # Rearrange columns to match names order
    result3 = result2[
        sorted(result2.columns, key=lambda t, s=s: s.fields.index(t))
    ]
    return result3


@dispatch(Expr, (DataFrame, DaskDataFrame))
def post_compute_by(t, df):
    return df.reset_index(drop=True)


@dispatch((Summary, Reduction), (DataFrame, DaskDataFrame))
def post_compute_by(t, df):
    return df.reset_index()


@dispatch(By, NDFrame)
def compute_up(t, df, **kwargs):
    grouper = get_grouper(t, t.grouper, df)
    result = compute_by(t, t.apply, grouper, df)
    result2 = post_compute_by(t.apply, into(DataFrame, result))
    if isinstance(result2, DataFrame):
        result2.columns = t.fields
    return result2


def concat_nodup(a, b):
    """ Concatenate two dataframes/series without duplicately named columns


    >>> df = DataFrame([[1, 'Alice',   100],
    ...                 [2, 'Bob',    -200],
    ...                 [3, 'Charlie', 300]],
    ...                columns=['id','name', 'amount'])

    >>> concat_nodup(df, df)
       id     name  amount
    0   1    Alice     100
    1   2      Bob    -200
    2   3  Charlie     300


    >>> concat_nodup(df.name, df.amount)
          name  amount
    0    Alice     100
    1      Bob    -200
    2  Charlie     300



    >>> concat_nodup(df, df.amount + df.id)
       id     name  amount    0
    0   1    Alice     100  101
    1   2      Bob    -200 -198
    2   3  Charlie     300  303
    """

    if isinstance(a, DataFrame) and isinstance(b, DataFrame):
        return pd.concat([a, b[[c for c in b.columns if c not in a.columns]]],
                         axis=1)
    if isinstance(a, DataFrame) and isinstance(b, Series):
        if b.name not in a.columns:
            return pd.concat([a, b], axis=1)
        else:
            return a
    if isinstance(a, Series) and isinstance(b, DataFrame):
        return pd.concat([a, b[[c for c in b.columns if c != a.name]]], axis=1)
    if isinstance(a, Series) and isinstance(b, Series):
        if a.name == b.name:
            return a
        else:
            return pd.concat([a, b], axis=1)


pdsort = getattr(
    pd.DataFrame,
    'sort' if LooseVersion(pd.__version__) < '0.17.0' else 'sort_values'
)


@dispatch(Sort, (DataFrame, DaskDataFrame))
def compute_up(t, df, **kwargs):
    return pdsort(df, t.key, ascending=t.ascending)


@dispatch(Sort, Series)
def compute_up(t, s, **kwargs):
    try:
        return s.sort_values(ascending=t.ascending)
    except AttributeError:
        return s.order(ascending=t.ascending)


@dispatch(Sample, (Series, DataFrame))
def compute_up(t, df, **kwargs):
    from math import modf
    if t.frac is not None:
        # Work around annoying edge case: Python's round() builtin (which
        # Pandas' sample() uses) rounds 0.5, 2.5, 4.5, ... down to 0, 2, 4, ...,
        # while it rounds 1.5, 3.5, 5.5, ... up.  This is inconsistent with any
        # sane implementation of floating point rounding, including SQL's, so
        # we work around it here.
        fractional, integral = modf(t.frac * df.shape[0])
        n = int(integral + (0 if fractional < 0.5 else 1))
    else:
        n = min(t.n, df.shape[0])
    return df.sample(n=n)


@dispatch(Sample, (DaskDataFrame, DaskSeries))
def compute_up(t, df, **kwargs):
    # Dask doesn't support sample(n=...), only sample(frac=...), so we have a
    # separate dispatch for dask objects.
    if t.frac is not None:
        frac = t.frac
    else:
        nrows = len(df)
        frac = float(min(t.n, nrows)) / nrows
    return df.sample(frac=frac)


@dispatch(Head, (Series, DataFrame, DaskDataFrame, DaskSeries))
def compute_up(t, df, **kwargs):
    return df.head(t.n)


@dispatch(Tail, (Series, DataFrame, DaskDataFrame, DaskSeries))
def compute_up(t, df, **kwargs):
    return df.tail(t.n)


@dispatch(Label, DataFrame)
def compute_up(t, df, **kwargs):
    return type(df)(df, columns=[t.label])


@dispatch(Label, Series)
def compute_up(t, df, **kwargs):
    return Series(df, name=t.label)


@dispatch(ReLabel, (DataFrame, DaskDataFrame))
def compute_up(t, df, **kwargs):
    return df.rename(columns=dict(t.labels))


@dispatch(ReLabel, (Series, DaskSeries))
def compute_up(t, s, **kwargs):
    labels = t.labels
    if len(labels) > 1:
        raise ValueError('You can only relabel a Series with a single name')
    pair, = labels
    _, replacement = pair
    return Series(s, name=replacement)


@dispatch(Map, DataFrame)
def compute_up(t, df, **kwargs):
    return df.apply(lambda tup: t.func(*tup), axis=1)


@dispatch(Map, Series)
def compute_up(t, df, **kwargs):
    result = df.map(t.func)
    try:
        result.name = t._name
    except NotImplementedError:
        # We don't have a schema, but we should still be able to map
        result.name = df.name
    return result


@dispatch(Apply, (Series, DataFrame))
def compute_up(t, df, **kwargs):
    return t.func(df)


@dispatch(Merge, NDFrame)
def compute_up(t, df, scope=None, **kwargs):
    subexpression = common_subexpression(*t.children)
    scope = merge_dicts(scope or {}, {subexpression: df})
    children = [
        compute(_child, scope, return_type='native')
        for _child in t.children
    ]
    return pd.concat(children, axis=1)


@dispatch(Summary, (DataFrame, DaskDataFrame))
def compute_up(expr, data, **kwargs):
    values = [
        compute(val, {expr._child: data}, return_type='native')
        for val in expr.values
    ]
    if expr.keepdims:
        return type(data)([values], columns=expr.fields)
    else:
        return Series(dict(zip(expr.fields, values)))


@dispatch(Summary, (Series, DaskSeries))
def compute_up(expr, data, **kwargs):
    result = tuple(
        compute(val, {expr._child: data}, return_type='native')
        for val in expr.values
    )
    if expr.keepdims:
        result = [result]
    return result


@dispatch(Like, Series)
def compute_up(expr, data, **kwargs):
    return data.str.contains(r'^%s$' % fnmatch.translate(expr.pattern))


def get_date_attr(s, attr, name):
    try:
        result = getattr(s.dt, attr)  # new in pandas 0.15
    except AttributeError:
        result = getattr(pd.DatetimeIndex(s), attr)
    result.name = name
    return result


@dispatch(DateTime, Series)
def compute_up(expr, s, **kwargs):
    return get_date_attr(s, expr.attr, expr._name)


@dispatch(UTCFromTimestamp, Series)
def compute_up(expr, s, **kwargs):
    return pd.datetools.to_datetime(s * 1e9, utc=True)


@dispatch(Millisecond, Series)
def compute_up(expr, s, **kwargs):
    return get_date_attr(s, 'microsecond',
                         '%s_millisecond' % expr._child._name) // 1000


@dispatch(Slice, (DataFrame, Series))
def compute_up(expr, df, **kwargs):
    index = expr.index
    if isinstance(index, tuple) and len(index) == 1:
        index = index[0]
    if isinstance(index, _inttypes + (list,)):
        return df.iloc[index]
    elif isinstance(index, slice):
        if index.stop is not None:
            return df.iloc[index.start:index.stop:index.step]
        else:
            return df.iloc[index]
    else:
        raise NotImplementedError()


@dispatch(count, DataFrame)
def compute_up(expr, df, **kwargs):
    result = df.shape[0]
    if expr.keepdims:
        result = Series([result], name=expr._name)
    return result


@dispatch(nelements, (DataFrame, Series))
def compute_up(expr, df, **kwargs):
    return df.shape[0]


@dispatch((count, nelements), (DaskDataFrame, DaskSeries))
def compute_up(expr, df, **kwargs):
    warnings.warn("Counting the elements of a dask object can be slow.")
    result = len(df)
    if expr.keepdims:
        result = DaskSeries([result], name=expr._name)
    return result


@dispatch(DateTimeTruncate, Series)
def compute_up(expr, data, **kwargs):
    return Series(compute_up(expr, into(np.ndarray, data), **kwargs),
                  name=expr._name)


@dispatch(IsIn, (Series, DaskSeries))
def compute_up(expr, data, **kwargs):
    return data.isin(expr._keys)


@dispatch(Coerce, (Series, DaskSeries))
def compute_up(expr, data, **kwargs):
    return data.astype(to_numpy_dtype(expr.schema))


@dispatch(Shift, Series)
def compute_up(expr, data, **kwargs):
    return data.shift(expr.n)


def array_coalesce(expr, lhs, rhs, wrap=None, **kwargs):
    res = np.where(pd.isnull(lhs), rhs, lhs)
    if not expr.dshape.shape:
        res = res.item()
    elif wrap:
        res = wrap(res)
    return res


@compute_up.register(
    Coalesce, (Series, DaskSeries), (np.ndarray, Series, base, DaskSeries)
)
@compute_up.register(
    Coalesce,
    (Series, base, DaskSeries), (np.ndarray, Series, DaskSeries)
)
def compute_up_coalesce(expr, lhs, rhs, **kwargs):
    return array_coalesce(expr, lhs, rhs, type(lhs))



@dispatch(Coalesce, (Series, DaskSeries, base))
def compute_up(t, data, **kwargs):
    if isinstance(t.lhs, Expr):
        lhs = data
        rhs = t.rhs
    else:
        lhs = t.lhs
        rhs = data

    return compute_up_coalesce(t, lhs, rhs)
