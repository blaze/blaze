"""

>>> from blaze import *

>>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
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
from sqlalchemy.sql.elements import ClauseElement
from operator import and_
from datashape import Record
from copy import copy

from ..dispatch import dispatch
from ..expr import Projection, Selection, Column, ColumnWise
from ..expr import BinOp, UnaryOp, USub, Join, mean, var, std, Reduction, count
from ..expr import nunique, Distinct, By, Sort, Head, Label, ReLabel, Merge
from ..expr import common_subexpression, Union, Summary, Like
from ..compatibility import reduce
from ..utils import unique
from .core import compute_up, compute, base
from ..data.utils import listpack

__all__ = ['sqlalchemy', 'select']


@dispatch(Projection, Selectable)
def compute_up(t, s, scope=None, **kwargs):
    # Walk up the tree to get the original columns
    if scope is None:
        scope = {}
    subexpression = t
    while hasattr(subexpression, 'child'):
        subexpression = subexpression.child
    csubexpression = compute(subexpression, scope)
    # Hack because csubexpression may be SQL object
    if not isinstance(csubexpression, Selectable):
        csubexpression = csubexpression.table
    columns = [csubexpression.c.get(col) for col in t.columns]

    return select(s).with_only_columns(columns)


@dispatch((Column, Projection), Select)
def compute_up(t, s, **kwargs):
    cols = [lower_column(s.c.get(col)) for col in t.columns]
    return s.with_only_columns(cols)


@dispatch(Column, sqlalchemy.Table)
def compute_up(t, s, **kwargs):
    return s.c.get(t.column)


@dispatch(ColumnWise, Select)
def compute_up(t, s, **kwargs):
    columns = [t.child[c] for c in t.child.columns]
    d = dict((t.child[c].scalar_symbol, lower_column(s.c.get(c)))
                    for c in t.child.columns)
    result = compute(t.expr, d)

    s = copy(s)
    s.append_column(result)
    return s.with_only_columns([result])


@dispatch(ColumnWise, Selectable)
def compute_up(t, s, **kwargs):
    columns = [t.child[c] for c in t.child.columns]
    d = dict((t.child[c].scalar_symbol, lower_column(s.c.get(c)))
                    for c in t.child.columns)
    return compute(t.expr, d)


@dispatch(BinOp, ClauseElement, (ClauseElement, base))
def compute_up(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(BinOp, (ClauseElement, base), ClauseElement)
def compute_up(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(UnaryOp, ClauseElement)
def compute_up(t, s, **kwargs):
    op = getattr(sa.func, t.symbol)
    return op(s)

@dispatch(USub, (sa.Column, Selectable))
def compute_up(t, s, **kwargs):
    return -s

@dispatch(Selection, Select)
def compute_up(t, s, **kwargs):
    predicate = compute_up(t.predicate, s)
    return s.where(predicate)


@dispatch(Selection, Selectable)
def compute_up(t, s, **kwargs):
    predicate = compute(t.predicate, {t.child: s})
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


@dispatch(Join, Selectable, Selectable)
def compute_up(t, lhs, rhs, **kwargs):
    condition = reduce(and_,
            [lower_column(lhs.c.get(l)) == lower_column(rhs.c.get(r))
        for l, r in zip(listpack(t.on_left), listpack(t.on_right))])

    if t.how == 'inner':
        join = lhs.join(rhs, condition)
        main, other = lhs, rhs
    elif t.how == 'left':
        join = lhs.join(rhs, condition, isouter=True)
        main, other = lhs, rhs
    elif t.how == 'right':
        join = rhs.join(lhs, condition, isouter=True)
        main, other = rhs, lhs
    else:
        # http://stackoverflow.com/questions/20361017/sqlalchemy-full-outer-join
        raise NotImplementedError("SQLAlchemy doesn't support full outer Join")

    columns = unique(list(main.columns) + list(other.columns),
                     key=lambda c: c.name)
    columns = (c for c in columns if c.name in t.columns)
    columns = sorted(columns, key=lambda c: t.columns.index(c.name))
    return select(list(columns)).select_from(join)



names = {mean: 'avg',
         var: 'variance',
         std: 'stdev'}

@dispatch((nunique, Reduction), Select)
def compute_up(t, s, **kwargs):
    columns = [t.child[c] for c in t.child.columns]
    d = dict((t.child[c], lower_column(s.c.get(c)))
                    for c in t.child.columns)
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

    if isinstance(t.child.schema[0], Record):
        name = list(t.child.schema[0].names)[0]
        result = result.label(name + '_' + type(t).__name__)

    return result


@dispatch(count, Selectable)
def compute_up(t, s, **kwargs):
    return compute_up(t, select(s), **kwargs)


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
        reductions = [compute(t.apply, {t.child: s})]
    elif isinstance(t.apply, Summary):
        reductions = [compute(val, {t.child: s}).label(name)
                for val, name in zip(t.apply.values, t.apply.names)]

    return sqlalchemy.select([grouper] + reductions).group_by(grouper)


@dispatch(By, ClauseElement)
def compute_up(t, s, **kwargs):
    if isinstance(t.grouper, Projection):
        grouper = [lower_column(s.c.get(col)) for col in t.grouper.columns]
    else:
        raise NotImplementedError("Grouper must be a projection, got %s"
                                  % t.grouper)
    if isinstance(t.apply, Reduction):
        reductions = [compute(t.apply, {t.child: s})]
    elif isinstance(t.apply, Summary):
        reductions = [compute(val, {t.child: s}).label(name)
                for val, name in zip(t.apply.values, t.apply.names)]

    return sqlalchemy.select(grouper + reductions).group_by(*grouper)


def lower_column(col):
    """ Return column from lower level tables if possible

    >>>
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
    if not isinstance(t.grouper, Projection):
        raise NotImplementedError("Grouper must be a projection, got %s"
                                  % t.grouper)

    if isinstance(t.apply, Reduction):
        reduction = compute(t.apply, {t.child: s})
        reductions = [reduction]

    elif isinstance(t.apply, Summary):
        reduction = compute(t.apply, {t.child: s})

    grouper = [lower_column(s.c.get(col)) for col in t.grouper.columns]
    s2 = reduction.group_by(*grouper)

    for g in grouper:
        s2.append_column(g)

    cols = s2._raw_columns
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
               for col, new_col in zip(t.child.columns, t.columns)]

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
def compute_up(t, s, **kwargs):
    columns = [t.child[c] for c in t.child.columns]
    d = dict((t.child[c], lower_column(s.c.get(c)))
                    for c in t.child.columns)

    cols = [compute(val, d).label(name)
                for name, val in zip(t.names, t.values)]

    s = copy(s)
    for c in cols:
        s.append_column(c)

    return s.with_only_columns(cols)


@dispatch(Summary, ClauseElement)
def compute_up(t, s, **kwargs):
    return select([compute(value, {t.child: s}).label(name)
        for value, name in zip(t.values, t.names)])


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
