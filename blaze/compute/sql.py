"""

>>> from blaze import *

>>> accounts = Symbol('accounts', 'var * {name: string, amount: int}')
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
import sqlalchemy as sa
import sqlalchemy
from sqlalchemy import sql
from sqlalchemy.sql import Selectable, Select
from sqlalchemy.sql.elements import ClauseElement, ColumnElement
from operator import and_
import itertools
from datashape import Record
from copy import copy
import toolz
from multipledispatch import MDNotImplementedError

from ..dispatch import dispatch
from ..expr import Projection, Selection, Field, Broadcast, Expr
from ..expr import BinOp, UnaryOp, USub, Join, mean, var, std, Reduction, count
from ..expr import nunique, Distinct, By, Sort, Head, Label, ReLabel, Merge
from ..expr import common_subexpression, Union, Summary, Like
from ..compatibility import reduce
from ..utils import unique
from .core import compute_up, compute, base
from ..data.utils import listpack

__all__ = ['sqlalchemy', 'select']



def inner_columns(s):
    if isinstance(s, sqlalchemy.Table):
        return s.c
    if isinstance(s, Selectable):
        return s.inner_columns
    raise NotImplementedError()

@dispatch(Projection, Selectable)
def compute_up(t, s, scope=None, **kwargs):
    # Walk up the tree to get the original columns
    if scope is None:
        scope = {}
    subexpression = t
    while hasattr(subexpression, '_child'):
        subexpression = subexpression._child
    csubexpression = compute(subexpression, scope)
    # Hack because csubexpression may be SQL object
    if not isinstance(csubexpression, Selectable):
        csubexpression = csubexpression.table
    columns = [csubexpression.c.get(col) for col in t.fields]

    return select(s).with_only_columns(columns)


@dispatch((Field, Projection), Select)
def compute_up(t, s, **kwargs):
    cols = list(s.inner_columns)
    cols = [lower_column(cols[t._child.fields.index(c)]) for c in t.fields]
    return s.with_only_columns(cols)


@dispatch(Field, sqlalchemy.Table)
def compute_up(t, s, **kwargs):
    return s.c.get(t._name)


@dispatch(Broadcast, Select)
def compute_up(t, s, **kwargs):
    d = dict((t._scalars[0][c], list(inner_columns(s))[i])
             for i, c in enumerate(t._scalars[0].fields))
    result = compute(t._scalar_expr, d)

    s = copy(s)
    s.append_column(result)
    return s.with_only_columns([result])


@dispatch(Broadcast, Selectable)
def compute_up(t, s, **kwargs):
    if len(t._children) != 1:
        raise ValueError()
    d = dict((t._scalars[0][c], list(inner_columns(s))[i])
             for i, c in enumerate(t._scalars[0].fields))
    return compute(t._scalar_expr, d)


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
    predicate = compute(t.predicate, toolz.merge(ns, scope))
    if isinstance(predicate, Select):
        predicate = list(list(predicate.columns)[0].base_columns)[0]
    return s.where(predicate)


@dispatch(Selection, Selectable)
def compute_up(t, s, scope=None, **kwargs):
    ns = dict((t._child[col.name], lower_column(col)) for col in s.columns)
    predicate = compute(t.predicate, toolz.merge(ns, scope))
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


@dispatch(Select, Selectable)
def _join_selectables(a, b, condition=None, **kwargs):
    if len(a.froms) > 1:
        raise MDNotImplementedError()

    return a.replace_selectable(a.froms[0],
                a.froms[0].join(b, condition, **kwargs))


@dispatch(Selectable, Select)
def _join_selectables(a, b, condition=None, **kwargs):
    if len(b.froms) > 1:
        raise MDNotImplementedError()
    return b.replace_selectable(b.froms[0],
                a.join(b.froms[0], condition, **kwargs))

@dispatch(Selectable, Selectable)
def _join_selectables(a, b, condition=None, **kwargs):
    return a.join(b, condition, **kwargs)


@dispatch(Join, Selectable, Selectable)
def compute_up(t, lhs, rhs, **kwargs):
    if name(lhs) == name(rhs):
        lhs = lhs.alias('%s_left' % name(lhs))
        rhs = rhs.alias('%s_right' % name(rhs))

    if isinstance(lhs, Select):
        ldict = dict((c.name, c) for c in lhs.inner_columns)
    else:
        ldict = lhs.c
    if isinstance(rhs, Select):
        rdict = dict((c.name, c) for c in rhs.inner_columns)
    else:
        rdict = rhs.c


    condition = reduce(and_,
            [lower_column(ldict.get(l)) == lower_column(rdict.get(r))
        for l, r in zip(listpack(t.on_left), listpack(t.on_right))])

    if t.how == 'inner':
        join = _join_selectables(lhs, rhs, condition=condition)
        main, other = lhs, rhs
    elif t.how == 'left':
        join = _join_selectables(lhs, rhs, condition=condition, isouter=True)
        main, other = lhs, rhs
    elif t.how == 'right':
        join = _join_selectables(rhs, lhs, condition=condition, isouter=True)
        main, other = rhs, lhs
    else:
        # http://stackoverflow.com/questions/20361017/sqlalchemy-full-outer-join
        raise NotImplementedError("SQLAlchemy doesn't support full outer Join")

    def cols(x):
        if isinstance(x, Select):
            return list(x.inner_columns)
        else:
            return list(x.columns)
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
    col = compute(t, d)

    s = copy(s)
    s.append_column(col)
    return s.with_only_columns([col])


@dispatch(Distinct, sqlalchemy.Column)
def compute_up(t, s, **kwargs):
    return s.distinct()


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

@dispatch(count, Select)
def compute_up(t, s, **kwargs):
    try:
        c = lower_column(list(s.primary_key)[0])
    except IndexError:
        c = lower_column(list(s.columns)[0])

    col = sqlalchemy.func.count(c)

    s = copy(s)
    s.append_column(col)
    return s.with_only_columns([col])


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
        reductions = [compute(t.apply, {t._child: s})]
    elif isinstance(t.apply, Summary):
        reductions = [compute(val, {t._child: s}).label(name)
                for val, name in zip(t.apply.values, t.apply.fields)]

    return sqlalchemy.select([grouper] + reductions).group_by(grouper)


@dispatch(By, ClauseElement)
def compute_up(t, s, **kwargs):
    if isinstance(t.grouper, (Field, Projection)):
        grouper = [lower_column(s.c.get(col)) for col in t.grouper.fields]
    else:
        raise NotImplementedError("Grouper must be a projection, got %s"
                                  % t.grouper)
    if isinstance(t.apply, Reduction):
        reductions = [compute(t.apply, {t._child: s})]
    elif isinstance(t.apply, Summary):
        reductions = [compute(val, {t._child: s}).label(name)
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


@dispatch(By, Select)
def compute_up(t, s, **kwargs):
    if not isinstance(t.grouper, (Field, Projection)):
        raise NotImplementedError("Grouper must be a projection, got %s"
                                  % t.grouper)

    if isinstance(t.apply, Reduction):
        reduction = compute(t.apply, {t._child: s})

    elif isinstance(t.apply, Summary):
        reduction = compute(t.apply, {t._child: s})

    grouper = [lower_column(s.c.get(col)) for col in t.grouper.fields]
    s2 = reduction.group_by(*grouper)

    for g in grouper:
        s2.append_column(g)

    cols = list(s2.inner_columns)
    cols = cols[-len(grouper):] + cols[:-len(grouper)]
    return s2.with_only_columns(cols)


@dispatch(Sort, Selectable)
def compute_up(t, s, **kwargs):
    if isinstance(t.key, (tuple, list)):
        raise NotImplementedError("Multi-column sort not yet implemented")
    col = getattr(s.c, t.key)
    if not t.ascending:
        col = sqlalchemy.desc(col)
    return select(s).order_by(col)


@dispatch(Head, Select)
def compute_up(t, s, **kwargs):
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
    children = [compute(child, {subexpression: s}) for child in t.children]
    return select(children)


@dispatch(Union, Selectable, tuple)
def compute_up(t, _, children):
    return sqlalchemy.union(*children)


@dispatch(Summary, Select)
def compute_up(t, s, scope=None, **kwargs):
    d = dict((t._child[c], list(inner_columns(s))[i])
            for i, c in enumerate(t._child.fields))

    cols = [compute(val, toolz.merge(scope, d)).label(name)
                for name, val in zip(t.fields, t.values)]

    s = copy(s)
    for c in cols:
        s.append_column(c)

    return s.with_only_columns(cols)


@dispatch(Summary, ClauseElement)
def compute_up(t, s, **kwargs):
    return select([compute(value, {t._child: s}).label(name)
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
