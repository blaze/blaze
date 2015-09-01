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

import itertools
from itertools import chain

from operator import and_, eq, attrgetter
from copy import copy

import sqlalchemy as sa

from sqlalchemy import sql, Table, MetaData
from sqlalchemy.sql import Selectable, Select, functions as safuncs
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.elements import ClauseElement, ColumnElement, ColumnClause
from sqlalchemy.sql.selectable import FromClause, ScalarSelect
from sqlalchemy.engine import Engine

import toolz

from toolz import unique, concat, pipe, first
from toolz.compatibility import zip
from toolz.curried import map

import numpy as np
import numbers

import warnings

from multipledispatch import MDNotImplementedError

from odo.backends.sql import metadata_of_engine, dshape_to_alchemy

from datashape.predicates import iscollection, isscalar, isrecord

from ..dispatch import dispatch

from .core import compute_up, compute, base


from ..expr import (
    Projection, Selection, Field, Broadcast, Expr, IsIn, Slice, BinOp, UnaryOp,
    Join, mean, var, std, Reduction, count, FloorDiv, UnaryStringFunction,
    strlen, DateTime, Coerce, nunique, Distinct, By, Sort, Head, Label, Concat,
    ReLabel, Merge, common_subexpression, Summary, Like, nelements, notnull
)

from ..expr.broadcast import broadcast_collect
from ..expr.math import isnan

from ..compatibility import reduce

from ..utils import listpack


__all__ = ['sa', 'select']


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
    cols = list(inner_columns(s))
    d = dict((t._scalars[0][c], cols[i])
             for i, c in enumerate(t._scalars[0].fields))
    result = compute(t._scalar_expr, d, post_compute=False).label(t._name)

    s = copy(s)
    s.append_column(result)
    return s.with_only_columns([result])


@dispatch(Broadcast, Selectable)
def compute_up(t, s, **kwargs):
    cols = list(inner_columns(s))
    d = dict((t._scalars[0][c], cols[i])
             for i, c in enumerate(t._scalars[0].fields))
    return compute(t._scalar_expr, d, post_compute=False).label(t._name)


@dispatch(Concat, (Select, Selectable), (Select, Selectable))
def compute_up(t, lhs, rhs, **kwargs):
    if t.axis != 0:
        raise ValueError(
            'Cannot concat along a non-zero axis in sql; perhaps you want'
            " 'merge'?",
        )

    return select(lhs).union_all(select(rhs)).alias()


@dispatch(Broadcast, sa.Column)
def compute_up(t, s, **kwargs):
    expr = t._scalar_expr
    return compute(expr, s, post_compute=False).label(expr._name)


@dispatch(BinOp, ColumnElement)
def compute_up(t, data, **kwargs):
    if isinstance(t.lhs, Expr):
        return t.op(data, t.rhs)
    else:
        return t.op(t.lhs, data)


@dispatch(BinOp, Select)
def compute_up(t, data, **kwargs):
    assert len(data.c) == 1, \
        'Select cannot have more than a single column when doing arithmetic'
    column = first(data.inner_columns)
    if isinstance(t.lhs, Expr):
        return t.op(column, t.rhs)
    else:
        return t.op(t.lhs, column)


@compute_up.register(BinOp, (ColumnElement, base), ColumnElement)
@compute_up.register(BinOp, ColumnElement, (ColumnElement, base))
def binop_sql(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(FloorDiv, ColumnElement)
def compute_up(t, data, **kwargs):
    if isinstance(t.lhs, Expr):
        return sa.func.floor(data / t.rhs)
    else:
        return sa.func.floor(t.rhs / data)


@compute_up.register(FloorDiv, (ColumnElement, base), ColumnElement)
@compute_up.register(FloorDiv, ColumnElement, (ColumnElement, base))
def binop_sql(t, lhs, rhs, **kwargs):
    return sa.func.floor(lhs / rhs)


@dispatch(isnan, ColumnElement)
def compute_up(t, s, **kwargs):
    return s == float('nan')


@dispatch(UnaryOp, ColumnElement)
def compute_up(t, s, **kwargs):
    sym = t.symbol
    return getattr(t, 'op', getattr(safuncs, sym, getattr(sa.func, sym)))(s)


@dispatch(Selection, sa.sql.ColumnElement)
def compute_up(expr, data, scope=None, **kwargs):
    predicate = compute(expr.predicate, data, post_compute=False)
    return sa.select([data]).where(predicate)


@dispatch(Selection, Selectable)
def compute_up(t, s, scope=None, **kwargs):
    ns = dict((t._child[col.name], col)
              for col in getattr(s, 'inner_columns', s.columns))
    predicate = compute(t.predicate, toolz.merge(ns, scope),
                        optimize=False, post_compute=False)
    try:
        return s.where(predicate)
    except AttributeError:
        return select([s]).where(predicate)


def select(s):
    """ Permissive SQL select

    Idempotent sa.select

    Wraps input in list if neccessary
    """
    if not isinstance(s, sa.sql.Select):
        if not isinstance(s, (tuple, list)):
            s = [s]
        s = sa.select(s)
    return s


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


_getname = attrgetter('name')


def _clean_join_name(opposite_side_colnames, suffix, c):
    if c.name not in opposite_side_colnames:
        return c
    else:
        return c.label(c.name + suffix)


@dispatch(Join, ClauseElement, ClauseElement)
def compute_up(t, lhs, rhs, **kwargs):
    if isinstance(lhs, ColumnElement):
        lhs = select(lhs)
    if isinstance(rhs, ColumnElement):
        rhs = select(rhs)
    if name(lhs) == name(rhs):
        left_suffix, right_suffix = t.suffixes
        lhs = lhs.alias('%s%s' % (name(lhs), left_suffix))
        rhs = rhs.alias('%s%s' % (name(rhs), right_suffix))

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
        main = lhs
    elif t.how == 'left':
        main, other = lhs, rhs
        join = _join_selectables(lhs, rhs, condition=condition, isouter=True)
    elif t.how == 'right':
        join = _join_selectables(rhs, lhs, condition=condition, isouter=True)
        main = rhs
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
    left_cols = cols(lhs)
    left_names = set(map(_getname, left_cols))
    right_cols = cols(rhs)
    right_names = set(map(_getname, right_cols))

    left_suffix, right_suffix = t.suffixes
    fields = [
        f.replace(left_suffix, '').replace(right_suffix, '') for f in t.fields
    ]
    columns = [c for c in main_cols if c.name in t._on_left]
    columns += [_clean_join_name(right_names, left_suffix, c)
                for c in left_cols
                if c.name in fields and c.name not in t._on_left]
    columns += [_clean_join_name(left_names, right_suffix, c)
                for c in right_cols
                if c.name in fields and c.name not in t._on_right]

    if isinstance(join, Select):
        return join.with_only_columns(columns)
    else:
        return sa.select(columns, from_obj=join)


names = {
    mean: 'avg'
}


def reconstruct_select(columns, original, **kwargs):
    return sa.select(columns,
                     from_obj=kwargs.pop('from_obj', None),
                     whereclause=kwargs.pop('whereclause',
                                            getattr(original,
                                                    '_whereclause', None)),
                     bind=kwargs.pop('bind', original.bind),
                     distinct=kwargs.pop('distinct',
                                         getattr(original,
                                                 '_distinct', False)),
                     group_by=kwargs.pop('group_by',
                                         getattr(original,
                                                 '_group_by_clause', None)),
                     having=kwargs.pop('having',
                                       getattr(original, '_having', None)),
                     limit=kwargs.pop('limit',
                                      getattr(original, '_limit', None)),
                     offset=kwargs.pop('offset',
                                       getattr(original, '_offset', None)),
                     order_by=kwargs.pop('order_by',
                                         getattr(original,
                                                 '_order_by_clause', None)),
                     **kwargs)


@dispatch((nunique, Reduction), Select)
def compute_up(expr, data, **kwargs):
    if expr.axis != (0,):
        raise ValueError('axis not equal to 0 not defined for SQL reductions')
    data = data.cte(name=next(aliases))
    cols = list(inner_columns(data))
    d = dict((expr._child[c], cols[i])
             for i, c in enumerate(expr._child.fields))
    return select([compute(expr, d, post_compute=False)])


@dispatch(Distinct, ColumnElement)
def compute_up(t, s, **kwargs):
    return s.distinct(*t.on).label(t._name)


@dispatch(Distinct, Select)
def compute_up(t, s, **kwargs):
    return s.distinct(*t.on)


@dispatch(Distinct, Selectable)
def compute_up(t, s, **kwargs):
    return select(s).distinct(*t.on)


@dispatch(Reduction, ClauseElement)
def compute_up(t, s, **kwargs):
    if t.axis != (0,):
        raise ValueError('axis not equal to 0 not defined for SQL reductions')
    try:
        op = getattr(sa.sql.functions, t.symbol)
    except AttributeError:
        op = getattr(sa.sql.func, names.get(type(t), t.symbol))
    return op(s).label(t._name)


prefixes = {
    std: 'stddev',
    var: 'var'
}


@dispatch((std, var), sql.elements.ColumnElement)
def compute_up(t, s, **kwargs):
    if t.axis != (0,):
        raise ValueError('axis not equal to 0 not defined for SQL reductions')
    funcname = 'samp' if t.unbiased else 'pop'
    full_funcname = '%s_%s' % (prefixes[type(t)], funcname)
    return getattr(sa.func, full_funcname)(s).label(t._name)


@dispatch(count, Selectable)
def compute_up(t, s, **kwargs):
    return s.count()


@dispatch(count, sa.Table)
def compute_up(t, s, **kwargs):
    if t.axis != (0,):
        raise ValueError('axis not equal to 0 not defined for SQL reductions')
    try:
        c = list(s.primary_key)[0]
    except IndexError:
        c = list(s.columns)[0]

    return sa.func.count(c)


@dispatch(nelements, (Select, ClauseElement))
def compute_up(t, s, **kwargs):
    return compute_up(t._child.count(), s)


@dispatch(count, Select)
def compute_up(t, s, **kwargs):
    if t.axis != (0,):
        raise ValueError('axis not equal to 0 not defined for SQL reductions')
    al = next(aliases)
    try:
        s2 = s.alias(al)
        col = list(s2.primary_key)[0]
    except (KeyError, IndexError):
        s2 = s.alias(al)
        col = list(s2.columns)[0]

    result = sa.func.count(col)

    return select([list(inner_columns(result))[0].label(t._name)])


@dispatch(nunique, sa.Column)
def compute_up(t, s, **kwargs):
    if t.axis != (0,):
        raise ValueError('axis not equal to 0 not defined for SQL reductions')
    return sa.func.count(s.distinct())


@dispatch(nunique, Selectable)
def compute_up(expr, data, **kwargs):
    return select(data).distinct().alias(next(aliases)).count()


@dispatch(By, sa.Column)
def compute_up(expr, data, **kwargs):
    grouper = lower_column(data)
    app = expr.apply
    if isinstance(app, Reduction):
        reductions = [compute(app, data, post_compute=False)]
    elif isinstance(app, Summary):
        reductions = [compute(val, data, post_compute=None).label(name)
                      for val, name in zip(app.values, app.fields)]

    return sa.select([grouper] + reductions).group_by(grouper)


@dispatch(By, ClauseElement)
def compute_up(expr, data, **kwargs):
    if not valid_grouper(expr):
        raise TypeError("Grouper must have a non-nested record or one "
                        "dimensional collection datashape, "
                        "got %s of type %r with dshape %s" %
                        (expr.grouper, type(expr.grouper).__name__,
                         expr.dshape))
    grouper = get_inner_columns(compute(expr.grouper, data,
                                        post_compute=False))
    app = expr.apply
    reductions = [compute(val, data, post_compute=False).label(name)
                  for val, name in zip(app.values, app.fields)]

    return sa.select(grouper + reductions).group_by(*grouper)


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


def is_nested_record(measure):
    """Predicate for checking whether `measure` is a nested ``Record`` dshape

    Examples
    --------
    >>> from datashape import dshape
    >>> is_nested_record(dshape('{a: int32, b: int32}').measure)
    False
    >>> is_nested_record(dshape('{a: var * ?float64, b: ?string}').measure)
    True
    """
    if not isrecord(measure):
        raise TypeError('Input must be a Record type got %s of type %r' %
                        (measure, type(measure).__name__))
    return not all(isscalar(t) for t in measure.types)


def valid_grouper(expr):
    ds = expr.dshape
    measure = ds.measure
    return (iscollection(ds) and
            (isscalar(measure) or
             (isrecord(measure) and not is_nested_record(measure))))


def valid_reducer(expr):
    ds = expr.dshape
    measure = ds.measure
    return (not iscollection(ds) and
            (isscalar(measure) or
             (isrecord(measure) and not is_nested_record(measure))))


@dispatch(By, Select)
def compute_up(expr, data, **kwargs):
    if not valid_grouper(expr):
        raise TypeError("Grouper must have a non-nested record or one "
                        "dimensional collection datashape, "
                        "got %s of type %r with dshape %s" %
                        (expr.grouper, type(expr.grouper).__name__, expr.dshape))

    s = alias_it(data)

    if valid_reducer(expr.apply):
        reduction = compute(expr.apply, s, post_compute=False)
    else:
        raise TypeError('apply must be a Summary expression')

    grouper = get_inner_columns(compute(expr.grouper, s, post_compute=False))
    reduction_columns = pipe(reduction.inner_columns,
                             map(get_inner_columns),
                             concat)
    columns = list(unique(chain(grouper, reduction_columns)))
    if (not isinstance(s, sa.sql.selectable.Alias) or
            (hasattr(s, 'froms') and isinstance(s.froms[0],
                                                sa.sql.selectable.Join))):
        assert len(s.froms) == 1, 'only a single FROM clause supported for now'
        from_obj, = s.froms
    else:
        from_obj = None

    return reconstruct_select(columns,
                              getattr(s, 'element', s),
                              from_obj=from_obj,
                              group_by=grouper)


@dispatch(Sort, (Selectable, Select))
def compute_up(t, s, **kwargs):
    s = select(s.alias())
    direction = sa.asc if t.ascending else sa.desc
    cols = [direction(lower_column(s.c[c])) for c in listpack(t.key)]
    return s.order_by(*cols)


@dispatch(Sort, (sa.Table, ColumnElement))
def compute_up(t, s, **kwargs):
    s = select(s)
    direction = sa.asc if t.ascending else sa.desc
    cols = [direction(lower_column(s.c[c])) for c in listpack(t.key)]
    return s.order_by(*cols)


@dispatch(Head, FromClause)
def compute_up(t, s, **kwargs):
    if s._limit is not None and s._limit <= t.n:
        return s
    return s.limit(t.n)


@dispatch(Head, sa.Table)
def compute_up(t, s, **kwargs):
    return s.select().limit(t.n)


@dispatch(Head, ColumnElement)
def compute_up(t, s, **kwargs):
    return sa.select([s]).limit(t.n)


@dispatch(Head, ScalarSelect)
def compute_up(t, s, **kwargs):
    return compute(t, s.element, post_compute=False)


@dispatch(Label, ColumnElement)
def compute_up(t, s, **kwargs):
    return s.label(t.label)


@dispatch(Label, FromClause)
def compute_up(t, s, **kwargs):
    assert len(s.c) == 1, \
        'expected %s to have a single column but has %d' % (s, len(s.c))
    inner_column, = s.inner_columns
    return reconstruct_select([inner_column.label(t.label)], s).as_scalar()


@dispatch(Expr, ScalarSelect)
def post_compute(t, s, **kwargs):
    return s.element


@dispatch(ReLabel, Selectable)
def compute_up(expr, data, **kwargs):
    names = data.c.keys()
    assert names == expr._child.fields
    d = dict(zip(names, getattr(data, 'inner_columns', data.c)))
    return sa.select(
        d[col].label(new_col) if col != new_col else d[col]
        for col, new_col in zip(expr._child.fields, expr.fields)
    )


@dispatch(FromClause)
def get_inner_columns(sel):
    try:
        return list(sel.inner_columns)
    except AttributeError:
        return list(map(lower_column, sel.c.values()))


@dispatch(ColumnElement)
def get_inner_columns(c):
    return [c]


@dispatch(ScalarSelect)
def get_inner_columns(sel):
    inner_columns = list(sel.inner_columns)
    assert len(inner_columns) == 1, 'ScalarSelect should have only ONE column'
    return list(map(lower_column, inner_columns))


@dispatch(sa.sql.functions.Function)
def get_inner_columns(f):
    unique_columns = unique(concat(map(get_inner_columns, f.clauses)))
    lowered = [x.label(getattr(x, 'name', None)) for x in unique_columns]
    return [getattr(sa.func, f.name)(*lowered)]


@dispatch(sa.sql.elements.Label)
def get_inner_columns(label):
    """
    Notes
    -----
    This should only ever return a list of length 1

    This is because we need to turn ScalarSelects into an actual column
    """
    name = label.name
    inner_columns = get_inner_columns(label.element)
    assert len(inner_columns) == 1
    return [lower_column(c).label(name) for c in inner_columns]


@dispatch(Select)
def get_all_froms(sel):
    return list(unique(sel.locate_all_froms()))


@dispatch(sa.Table)
def get_all_froms(t):
    return [t]


@dispatch(ColumnClause)
def get_all_froms(c):
    return [c.table]


def get_clause(data, kind):
    # arg SQLAlchemy doesn't allow things like data._group_by_clause or None
    assert kind == 'order_by' or kind == 'group_by', \
        'kind must be "order_by" or "group_by"'
    clause = getattr(data, '_%s_clause' % kind, None)
    return clause.clauses if clause is not None else None


@dispatch(Merge, (Selectable, Select, sa.Column))
def compute_up(expr, data, **kwargs):
    # get the common subexpression of all the children in the merge
    subexpression = common_subexpression(*expr.children)

    # compute each child, including the common subexpression
    children = [compute(child, {subexpression: data}, post_compute=False)
                for child in expr.children]

    # Get the original columns from the selection and rip out columns from
    # Selectables and ScalarSelects
    columns = list(unique(concat(map(get_inner_columns, children))))

    # we need these getattrs if data is a ColumnClause or Table
    from_obj = get_all_froms(data)
    assert len(from_obj) == 1, 'only a single FROM clause supported'
    return reconstruct_select(columns, data, from_obj=from_obj)


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
    scope = {t._child: s}
    return sa.select(
        compute(value, scope, post_compute=None).label(name)
        for value, name in zip(t.values, t.fields)
    )


@dispatch(Like, Selectable)
def compute_up(t, s, **kwargs):
    return compute_up(t, select(s), **kwargs)


@dispatch(Like, Select)
def compute_up(t, s, **kwargs):
    items = [(f.c.get(name), pattern.replace('*', '%'))
             for name, pattern in t.patterns.items()
             for f in s.froms if name in f.c]
    return s.where(reduce(and_, [key.like(pattern) for key, pattern in items]))


string_func_names = {
    # <blaze function name>: <SQL function name>
}


# TODO: remove if the alternative fix goes into PyHive
@compiles(sa.sql.functions.Function, 'hive')
def compile_char_length_on_hive(element, compiler, **kwargs):
    assert len(element.clauses) == 1, \
        'char_length must have a single clause, got %s' % list(element.clauses)
    if element.name == 'char_length':
        return compiler.visit_function(sa.func.length(*element.clauses),
                                       **kwargs)
    return compiler.visit_function(element, **kwargs)


@dispatch(strlen, ColumnElement)
def compute_up(expr, data, **kwargs):
    return sa.sql.functions.char_length(data).label(expr._name)


@dispatch(UnaryStringFunction, ColumnElement)
def compute_up(expr, data, **kwargs):
    func_name = type(expr).__name__
    func_name = string_func_names.get(func_name, func_name)
    return getattr(sa.sql.func, func_name)(data).label(expr._name)


@dispatch(notnull, ColumnElement)
def compute_up(expr, data, **kwargs):
    return data != None


@toolz.memoize
def table_of_metadata(metadata, name):
    if metadata.schema is not None:
        name = '.'.join((metadata.schema, name))
    if name not in metadata.tables:
        metadata.reflect(views=metadata.bind.dialect.supports_views)
    return metadata.tables[name]


def table_of_engine(engine, name):
    metadata = metadata_of_engine(engine)
    return table_of_metadata(metadata, name)


@dispatch(Field, sa.engine.Engine)
def compute_up(expr, data, **kwargs):
    return table_of_engine(data, expr._name)


@dispatch(DateTime, (ClauseElement, sa.sql.elements.ColumnElement))
def compute_up(expr, data, **kwargs):
    if expr.attr == 'date':
        return sa.func.date(data).label(expr._name)

    return sa.extract(expr.attr, data).label(expr._name)


@compiles(sa.sql.elements.Extract, 'hive')
def hive_extract_to_date_function(element, compiler, **kwargs):
    func = getattr(sa.func, element.field)(element.expr)
    return compiler.visit_function(func, **kwargs)


@compiles(sa.sql.elements.Extract, 'mssql')
def mssql_extract_to_datepart(element, compiler, **kwargs):
    func = sa.func.datepart(sa.sql.expression.column(element.field),
                            element.expr)
    return compiler.visit_function(func, **kwargs)


def engine_of(x):
    if isinstance(x, Engine):
        return x
    if isinstance(x, MetaData):
        return x.bind
    if isinstance(x, Table):
        return x.metadata.bind
    raise NotImplementedError("Can't deterimine engine of %s" % x)


@dispatch(Expr, ClauseElement)
def optimize(expr, _):
    return broadcast_collect(expr)


@dispatch(Field, sa.MetaData)
def compute_up(expr, data, **kwargs):
    return table_of_metadata(data, expr._name)


@dispatch(Expr, ClauseElement)
def post_compute(_, s, **kwargs):
    return select(s)


@dispatch(IsIn, ColumnElement)
def compute_up(expr, data, **kwargs):
    return data.in_(expr._keys)


@dispatch(Slice, (Select, Selectable, ColumnElement))
def compute_up(expr, data, **kwargs):
    index = expr.index[0]  # [0] replace_slices returns tuple ((start, stop), )
    if isinstance(index, slice):
        start = index.start or 0
        if start < 0:
            raise ValueError('start value of slice cannot be negative'
                             ' with a SQL backend')

        stop = index.stop
        if stop is not None and stop < 0:
            raise ValueError('stop value of slice cannot be negative with a '
                             'SQL backend.')

        if index.step is not None and index.step != 1:
            raise ValueError('step parameter in slice objects not supported '
                             'with SQL backend')

    elif isinstance(index, (np.integer, numbers.Integral)):
        if index < 0:
            raise ValueError('integer slice cannot be negative for the'
                             ' SQL backend')
        start = index
        stop = start + 1
    else:
        raise TypeError('type %r not supported for slicing wih SQL backend'
                        % type(index).__name__)

    warnings.warn('The order of the result set from a Slice expression '
                  'computed against the SQL backend is not deterministic.')

    if stop is None:  # Represents open-ended slice. e.g. [3:]
        return select(data).offset(start)
    else:
        return select(data).offset(start).limit(stop - start)


@dispatch(Coerce, ColumnElement)
def compute_up(expr, data, **kwargs):
    return sa.cast(data, dshape_to_alchemy(expr.to)).label(expr._name)
