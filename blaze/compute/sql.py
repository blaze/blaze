"""

>>> from blaze import *

>>> accounts = symbol('accounts', 'var * {name: string, amount: int}')
>>> deadbeats = accounts[accounts['amount'] < 0]['name']

>>> from sqlalchemy import Table, Column, MetaData, Integer, String
>>> t = Table('accounts', MetaData(),
...           Column('name', String, primary_key = True),
...           Column('amount', Integer))
>>> print(compute(deadbeats, t))  # doctest: +SKIP
SELECT accounts.name
FROM accounts
WHERE accounts.amount < :amount_1
"""
from __future__ import absolute_import, division, print_function

import operator
import sqlalchemy as sa
import sqlalchemy
from sqlalchemy import sql, Table, MetaData
from sqlalchemy.sql import Selectable, Select
from sqlalchemy.sql.elements import ClauseElement, ColumnElement
from sqlalchemy.engine import Engine
from operator import and_, eq
import itertools
from copy import copy
import toolz
from multipledispatch import MDNotImplementedError
from datashape.predicates import isscalar, isrecord
from odo.backends.sql import metadata_of_engine

from ..dispatch import dispatch
from ..expr import Projection, Selection, Field, Broadcast, Expr, Symbol
from ..expr import BinOp, UnaryOp, USub, Join, mean, var, std, Reduction, count
from ..expr import nunique, Distinct, By, Sort, Head, Label, ReLabel, Merge
from ..expr import common_subexpression, Summary, Like, nelements
from ..compatibility import reduce
from .core import compute_up, compute, base
from ..utils import listpack

__all__ = ['sqlalchemy', 'select']



def inner_columns(s):
    try:
        return s.inner_columns
    except AttributeError:
        return s.c
    raise TypeError()

@dispatch(Projection, Selectable)
def compute_up(t, s, scope=None, **kwargs):
    d = dict((c.name, c) for c in inner_columns(s))
    return select(s).with_only_columns([d[field] for field in t.fields])


@dispatch((Field, Projection), Select)
def compute_up(t, s, **kwargs):
    cols = list(s.inner_columns)
    cols = [lower_column(cols[t._child.fields.index(c)]) for c in t.fields]
    return s.with_only_columns(cols)


@dispatch(Field, ClauseElement)
def compute_up(t, s, **kwargs):
    return s.c.get(t._name)


@dispatch(Broadcast, Select)
def compute_up(t, s, **kwargs):
    d = dict((t._scalars[0][c], list(inner_columns(s))[i])
             for i, c in enumerate(t._scalars[0].fields))
    result = compute(t._scalar_expr, d, post_compute=False)

    s = copy(s)
    s.append_column(result)
    return s.with_only_columns([result])


@dispatch(Broadcast, Selectable)
def compute_up(t, s, **kwargs):
    d = dict((t._scalars[0][c], list(inner_columns(s))[i])
             for i, c in enumerate(t._scalars[0].fields))
    return compute(t._scalar_expr, d, post_compute=False)


@dispatch(BinOp, ColumnElement)
def compute_up(t, data, **kwargs):
    if isinstance(t.lhs, Expr):
        return t.op(data, t.rhs)
    else:
        return t.op(t.lhs, data)


@dispatch(BinOp, ColumnElement, (ColumnElement, base))
def compute_up(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(BinOp, (ColumnElement, base), ColumnElement)
def compute_up(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(UnaryOp, ColumnElement)
def compute_up(t, s, **kwargs):
    op = getattr(sa.func, t.symbol)
    return op(s)


@dispatch(USub, (sa.Column, Selectable))
def compute_up(t, s, **kwargs):
    return -s


@dispatch(Selection, Select)
def compute_up(t, s, scope=None, **kwargs):
    ns = dict((t._child[col.name], col) for col in s.inner_columns)
    predicate = compute(t.predicate, toolz.merge(ns, scope),
                            optimize=False, post_compute=False)
    if isinstance(predicate, Select):
        predicate = list(list(predicate.columns)[0].base_columns)[0]
    return s.where(predicate)


@dispatch(Selection, Selectable)
def compute_up(t, s, scope=None, **kwargs):
    ns = dict((t._child[col.name], lower_column(col)) for col in s.columns)
    predicate = compute(t.predicate, toolz.merge(ns, scope),
                            optimize=False, post_compute=False)
    if isinstance(predicate, Select):
        predicate = list(list(predicate.columns)[0].base_columns)[0]
    try:
        return s.where(predicate)
    except AttributeError:
        return select([s]).where(predicate)


def select(s):
    """ Permissive SQL select

    Idempotent sqlalchemy.select

    Wraps input in list if neccessary
    """
    if not isinstance(s, sqlalchemy.sql.Select):
        if not isinstance(s, (tuple, list)):
            s = [s]
        s = sa.select(s)
    return s


def computefull(t, s):
    return select(compute(t, s))

table_names = ('table_%d' % i for i in itertools.count(1))

def name(sel):
    """ Name of a selectable """
    if hasattr(sel, 'name'):
        return sel.name
    if hasattr(sel, 'froms'):
        if len(sel.froms) == 1:
            return name(sel.froms[0])
    return next(table_names)

@dispatch(Select, Select)
def _join_selectables(a, b, condition=None, **kwargs):
    return a.join(b, condition, **kwargs)


@dispatch(Select, ClauseElement)
def _join_selectables(a, b, condition=None, **kwargs):
    if len(a.froms) > 1:
        raise MDNotImplementedError()

    return a.replace_selectable(a.froms[0],
                a.froms[0].join(b, condition, **kwargs))


@dispatch(ClauseElement, Select)
def _join_selectables(a, b, condition=None, **kwargs):
    if len(b.froms) > 1:
        raise MDNotImplementedError()
    return b.replace_selectable(b.froms[0],
                a.join(b.froms[0], condition, **kwargs))

@dispatch(ClauseElement, ClauseElement)
def _join_selectables(a, b, condition=None, **kwargs):
    return a.join(b, condition, **kwargs)


@dispatch(Join, ClauseElement, ClauseElement)
def compute_up(t, lhs, rhs, **kwargs):
    if isinstance(lhs, ColumnElement):
        lhs = select(lhs)
    if isinstance(rhs, ColumnElement):
        rhs = select(rhs)
    if name(lhs) == name(rhs):
        lhs = lhs.alias('%s_left' % name(lhs))
        rhs = rhs.alias('%s_right' % name(rhs))

    lhs = alias_it(lhs)
    rhs = alias_it(rhs)

    if isinstance(lhs, Select):
        lhs = lhs.alias(next(aliases))
        left_conds = [lhs.c.get(c) for c in listpack(t.on_left)]
    else:
        ldict = dict((c.name, c) for c in inner_columns(lhs))
        left_conds = [ldict.get(c) for c in listpack(t.on_left)]

    if isinstance(rhs, Select):
        rhs = rhs.alias(next(aliases))
        right_conds = [rhs.c.get(c) for c in listpack(t.on_right)]
    else:
        rdict = dict((c.name, c) for c in inner_columns(rhs))
        right_conds = [rdict.get(c) for c in listpack(t.on_right)]

    condition = reduce(and_, map(eq, left_conds, right_conds))

    # Perform join
    if t.how == 'inner':
        join = _join_selectables(lhs, rhs, condition=condition)
        main, other = lhs, rhs
    elif t.how == 'left':
        main, other = lhs, rhs
        join = _join_selectables(lhs, rhs, condition=condition, isouter=True)
    elif t.how == 'right':
        join = _join_selectables(rhs, lhs, condition=condition, isouter=True)
        main, other = rhs, lhs
    else:
        # http://stackoverflow.com/questions/20361017/sqlalchemy-full-outer-join
        raise ValueError("SQLAlchemy doesn't support full outer Join")

    """
    We now need to arrange the columns in the join to match the columns in
    the expression.  We care about order and don't want repeats
    """
    if isinstance(join, Select):
        def cols(x):
            if isinstance(x, Select):
                return list(x.inner_columns)
            else:
                return list(x.columns)
    else:
        cols = lambda x: list(x.columns)

    main_cols = cols(main)
    other_cols = cols(other)
    left_cols = cols(lhs)
    right_cols = cols(rhs)

    fields = [f.replace('_left', '').replace('_right', '') for f in t.fields]
    columns = [c for c in main_cols if c.name in t._on_left]
    columns += [c for c in left_cols if c.name in fields
                                    and c.name not in t._on_left]
    columns += [c for c in right_cols if c.name in fields
                                     and c.name not in t._on_right]

    if isinstance(join, Select):
        return join.with_only_columns(columns)
    else:
        return sqlalchemy.sql.select(columns, from_obj=join)


names = {mean: 'avg',
         var: 'variance',
         std: 'stdev'}


@dispatch((nunique, Reduction), Select)
def compute_up(t, s, **kwargs):
    d = dict((t._child[c], list(inner_columns(s))[i])
            for i, c in enumerate(t._child.fields))
    col = compute(t, d, post_compute=False)

    s = copy(s)
    s.append_column(col)
    return s.with_only_columns([col])


@dispatch(Distinct, sqlalchemy.Column)
def compute_up(t, s, **kwargs):
    return s.distinct().label(t._name)


@dispatch(Distinct, Select)
def compute_up(t, s, **kwargs):
    return s.distinct()


@dispatch(Reduction, sql.elements.ClauseElement)
def compute_up(t, s, **kwargs):
    try:
        op = getattr(sqlalchemy.sql.functions, t.symbol)
    except AttributeError:
        symbol = names.get(type(t), t.symbol)
        op = getattr(sqlalchemy.sql.func, symbol)
    result = op(s)

    return result.label(t._name)


@dispatch(count, Selectable)
def compute_up(t, s, **kwargs):
    return compute_up(t, select(s), **kwargs)


@dispatch(count, sqlalchemy.Table)
def compute_up(t, s, **kwargs):
    try:
        c = list(s.primary_key)[0]
    except IndexError:
        c = list(s.columns)[0]

    return sqlalchemy.sql.functions.count(c)


@dispatch(nelements, (Select, ClauseElement))
def compute_up(t, s, **kwargs):
    return compute_up(t._child.count(), s)


@dispatch(count, Select)
def compute_up(t, s, **kwargs):
    al = next(aliases)
    try:
        s2 = s.alias(al)
        col = list(s2.primary_key)[0]
    except (KeyError, IndexError):
        s2 = s.alias(al)
        col = list(s2.columns)[0]

    result = sqlalchemy.sql.functions.count(col)

    return select([list(inner_columns(result))[0].label(t._name)])


@dispatch(nunique, sqlalchemy.Column)
def compute_up(t, s, **kwargs):
    return sqlalchemy.sql.functions.count(s.distinct())


@dispatch(Distinct, sqlalchemy.Table)
def compute_up(t, s, **kwargs):
    return select(s).distinct()

@dispatch(By, sqlalchemy.Column)
def compute_up(t, s, **kwargs):
    grouper = lower_column(s)
    if isinstance(t.apply, Reduction):
        reductions = [compute(t.apply, {t._child: s}, post_compute=False)]
    elif isinstance(t.apply, Summary):
        reductions = [compute(val, {t._child: s}, post_compute=None).label(name)
                for val, name in zip(t.apply.values, t.apply.fields)]

    return sqlalchemy.select([grouper] + reductions).group_by(grouper)


@dispatch(By, ClauseElement)
def compute_up(t, s, **kwargs):
    if (isinstance(t.grouper, (Field, Projection)) or
            t.grouper is t._child):
        # d = dict((c.name, c) for c in inner_columns(s))
        # grouper = [d[col] for col in t.grouper.fields]
        grouper = [lower_column(s.c.get(col)) for col in t.grouper.fields]
    else:
        raise ValueError("Grouper must be a projection, got %s"
                                  % t.grouper)
    if isinstance(t.apply, Reduction):
        reductions = [compute(t.apply, {t._child: s}, post_compute=False)]
    elif isinstance(t.apply, Summary):
        reductions = [compute(val, {t._child: s}, post_compute=None).label(name)
                for val, name in zip(t.apply.values, t.apply.fields)]

    return sqlalchemy.select(grouper + reductions).group_by(*grouper)


def lower_column(col):
    """ Return column from lower level tables if possible

    >>> metadata = sa.MetaData()

    >>> s = sa.Table('accounts', metadata,
    ...              sa.Column('name', sa.String),
    ...              sa.Column('amount', sa.Integer),
    ...              sa.Column('id', sa.Integer, primary_key=True),
    ...              )

    >>> s2 = select([s])
    >>> s2.c.amount is s.c.amount
    False

    >>> lower_column(s2.c.amount) is s.c.amount
    True

    >>> lower_column(s2.c.amount)
    Column('amount', Integer(), table=<accounts>)
    """

    old = None
    while col is not None and col is not old:
        old = col
        if not hasattr(col, 'table') or not hasattr(col.table, 'froms'):
            return col
        for f in col.table.froms:
            if f.corresponding_column(col) is not None:
                col = f.corresponding_column(col)

    return old


aliases = ('alias_%d' % i for i in itertools.count(1))

@toolz.memoize
def alias_it(s):
    """ Alias a Selectable if it has a group by clause """
    if (hasattr(s, '_group_by_clause') and
        s._group_by_clause is not None and
        len(s._group_by_clause)):
        return s.alias(next(aliases))
    else:
        return s


@dispatch(By, Select)
def compute_up(t, s, **kwargs):
    if not (isinstance(t.grouper, (Field, Projection))
            or t.grouper is t._child):
        raise ValueError("Grouper must be a projection, got %s"
                                  % t.grouper)

    s = alias_it(s)

    if isinstance(t.apply, Reduction):
        reduction = compute(t.apply, {t._child: s}, post_compute=False)

    elif isinstance(t.apply, Summary):
        reduction = compute(t.apply, {t._child: s}, post_compute=False)

    # d = dict((c.name, c) for c in inner_columns(s))
    # grouper = [d[col] for col in t.grouper.fields]
    grouper = [lower_column(s.c.get(col)) for col in t.grouper.fields]

    s2 = reduction.group_by(*grouper)

    for g in grouper:
        s2.append_column(g)

    cols = list(s2.inner_columns)
    cols = cols[-len(grouper):] + cols[:-len(grouper)]
    return s2.with_only_columns(cols)


@dispatch(Sort, ClauseElement)
def compute_up(t, s, **kwargs):
    if isinstance(t.key, (tuple, list)):
        raise ValueError("Multi-column sort not yet implemented")
    s = select(s)
    col = lower_column(getattr(s.c, t.key))
    if not t.ascending:
        col = sqlalchemy.desc(col)
    return s.order_by(col)


@dispatch(Head, Select)
def compute_up(t, s, **kwargs):
    if s._limit is not None and s._limit <= t.n:
        return s
    return s.limit(t.n)


@dispatch(Head, ClauseElement)
def compute_up(t, s, **kwargs):
    return select(s).limit(t.n)


@dispatch(Label, ClauseElement)
def compute_up(t, s, **kwargs):
    return s.label(t.label)


@dispatch(ReLabel, Selectable)
def compute_up(t, s, **kwargs):
    columns = [getattr(s.c, col).label(new_col)
               if col != new_col else
               getattr(s.c, col)
               for col, new_col in zip(t._child.fields, t.fields)]

    return select(columns)


@dispatch(Merge, Selectable)
def compute_up(t, s, **kwargs):
    subexpression = common_subexpression(*t.children)
    children = [compute(child, {subexpression: s}, post_compute=False)
                 for child in t.children]
    return select(children)


@dispatch(Summary, Select)
def compute_up(t, s, scope=None, **kwargs):
    d = dict((t._child[c], list(inner_columns(s))[i])
            for i, c in enumerate(t._child.fields))

    cols = [compute(val, toolz.merge(scope, d), post_compute=None).label(name)
                for name, val in zip(t.fields, t.values)]

    s = copy(s)
    for c in cols:
        s.append_column(c)

    return s.with_only_columns(cols)


@dispatch(Summary, ClauseElement)
def compute_up(t, s, **kwargs):
    return select([compute(value, {t._child: s}, post_compute=None).label(name)
        for value, name in zip(t.values, t.fields)])


@dispatch(Like, Selectable)
def compute_up(t, s, **kwargs):
    return compute_up(t, select(s), **kwargs)


@dispatch(Like, Select)
def compute_up(t, s, **kwargs):
    d = dict()
    for name, pattern in t.patterns.items():
        for f in s.froms:
            if f.c.has_key(name):
                d[f.c.get(name)] = pattern.replace('*', '%')

    return s.where(reduce(and_,
                          [key.like(pattern) for key, pattern in d.items()]))


@toolz.memoize
def table_of_metadata(metadata, name):
    if name not in metadata.tables:
        metadata.reflect(views=metadata.bind.dialect.supports_views)
    return metadata.tables[name]


def table_of_engine(engine, name):
    metadata = metadata_of_engine(engine)
    return table_of_metadata(metadata, name)


@dispatch(Field, sqlalchemy.engine.Engine)
def compute_up(expr, data, **kwargs):
    return table_of_engine(data, expr._name)


def engine_of(x):
    if isinstance(x, Engine):
        return x
    if isinstance(x, MetaData):
        return x.bind
    if isinstance(x, Table):
        return x.metadata.bind
    raise NotImplementedError("Can't deterimine engine of %s" % x)


from ..expr.broadcast import broadcast_collect


@dispatch(Expr, sa.sql.elements.ClauseElement)
def optimize(expr, _):
    return broadcast_collect(expr)


@dispatch(Field, sqlalchemy.MetaData)
def compute_up(expr, data, **kwargs):
    return table_of_metadata(data, expr._name)


@dispatch(Expr, sa.sql.elements.ClauseElement)
def post_compute(_, s, **kwargs):
    return select(s)
