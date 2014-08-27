from __future__ import absolute_import, print_function

import numbers

from ..expr import TableSymbol, Sort, Head, Distinct, Expr, Projection, By, Map
from ..expr import Selection, Relational, ScalarSymbol, ColumnWise, Summary
from ..expr import count, sum, min, max, mean, nunique, var, std
from ..expr import Arithmetic, UnaryOp

from ..compatibility import basestring

from ..dispatch import dispatch

from cytoolz import take, first

import rethinkdb as rt
from rethinkdb.ast import RqlQuery, Group as RqlGroup


__all__ = ['compute_one', 'discover', 'RTable', 'create_index', 'drop']


class RTable(object):

    """Lightweight wrapper around ``rethinkdb.ast.Table``.

    Parameters
    ----------
    t : RqlQuery
        Usually an instance of ``rethinkdb.ast.Table``
    conn : rethinkdb.net.Connection
        A connection object against which queries will run.

    Examples
    --------
    >>> import rethinkdb as r
    >>> t = RTable(r.table('test'), r.connect())
    >>> t
    RTable(t=r.table('test'))
    """

    def __init__(self, t, conn):
        self.t = t
        self.conn = conn

    def __repr__(self):
        return '%s(t=%s)' % (type(self).__name__, self.t)

    __str__ = __repr__


@dispatch(RTable)
def discover(t, n=50):
    return discover(list(take(n, t.t.run(t.conn))))


@dispatch(Expr, RTable)
def compute_one(e, t, **kwargs):
    return compute_one(e, t.t, **kwargs)


@dispatch(TableSymbol, RqlQuery)
def compute_one(_, t, **kwargs):
    return t


@dispatch(Projection, RqlQuery)
def compute_one(p, t, **kwargs):
    return t.pluck(*p.columns)


@dispatch(Head, RqlQuery)
def compute_one(h, t, **kwargs):
    return t.limit(h.n)


@dispatch(basestring)
def default_sort_order(key, f=rt.asc):
    return f(key),


@dispatch(list)
def default_sort_order(keys, f=rt.asc):
    for key in keys:
        yield f(key)


@dispatch(Sort, RqlQuery)
def compute_one(s, t, **kwargs):
    f = rt.asc if s.ascending else rt.desc
    return t.order_by(*default_sort_order(s.key, f=f))


@dispatch(sum, RqlQuery)
def compute_one(f, t, **kwargs):
    return t.sum(f.child.column)


@dispatch(var, RqlQuery)
def compute_one(v, t, **kwargs):
    column = v.child.column
    tavg = t.avg(column)
    row = rt.row[column]
    sumsqr = t.map(row * row).sum() / t.pluck(column).count()
    ms = tavg * tavg
    return sumsqr - ms


@dispatch(std, RqlQuery)
def compute_one(s, t, **kwargs):
    raise NotImplementedError('No way to call a function on a number')


@dispatch((min, max), RqlQuery)
def compute_one(f, t, **kwargs):
    column = f.child.column
    return getattr(t, type(f).__name__)(column)[column]


@dispatch(count, RqlQuery)
def compute_one(f, t, **kwargs):
    return t.count()


@dispatch(mean, RqlQuery)
def compute_one(f, t, **kwargs):
    return t.avg(f.child.column)


@dispatch(nunique, RqlQuery)
def compute_one(f, t, **kwargs):
    return t.distinct().count()


@dispatch(ScalarSymbol, RqlQuery)
def compute_one(ss, _, **kwargs):
    return rt.row[ss.name]


@dispatch((basestring, numbers.Real), RqlQuery)
def compute_one(s, _, **kwargs):
    return s


@dispatch((Relational, Arithmetic), RqlQuery)
def compute_one(r, t, **kwargs):
    return r.op(compute_one(r.lhs, t, **kwargs),
                compute_one(r.rhs, t, **kwargs))


@dispatch(UnaryOp, (RqlQuery, RTable))
def compute_one(o, t, **kwargs):
    raise NotImplementedError('ReQL does not support unary operations')


@dispatch(Selection, RqlQuery)
def compute_one(s, t, **kwargs):
    e = s.predicate.expr
    return t.filter(e.op(compute_one(e.lhs, t, **kwargs),
                         compute_one(e.rhs, t, **kwargs)))


@dispatch(Distinct, RqlQuery)
def compute_one(d, t, **kwargs):
    return t.with_fields(*d.columns).distinct()


@dispatch(ColumnWise, RqlQuery)
def compute_one(cw, t, **kwargs):
    return t.map(compute_one(cw.expr, t, **kwargs))


@dispatch(Summary, RqlQuery)
def compute_one(s, subgroup, **kwargs):
    return dict(('%s_%s' % (op.child.column, name),
                 compute_one(op, subgroup, **kwargs))
                for name, op in zip(s.names, s.values))


@dispatch(Summary, RqlGroup)
def compute_one(s, g, **kwargs):
    return g.do(lambda x: compute_one(s, x, **kwargs))


@dispatch(By, RqlQuery)
def compute_one(b, t, **kwargs):
    return compute_one(b.apply, t.group(*b.grouper.columns), **kwargs)


@dispatch(Map, RqlQuery)
def compute_one(m, t, **kwargs):
    return t.map(dict((column, m.func(rt.row[column])) for column in m.columns))


@dispatch(Expr, RTable, dict)
def post_compute(e, t, d):
    return post_compute(e, t.t, d)


@dispatch(Expr, RqlQuery, dict)
def post_compute(e, t, d):
    # TODO: test if we can use any connection for tables on different nodes
    assert len(d) == 1  # we only have a single RTable in scope
    return t.run(first(d.values()).conn)


@dispatch(RTable, basestring)
def create_index(t, column, **kwargs):
    t.t.index_create(column, **kwargs).run(t.conn)


@dispatch(RTable)
def drop(t):
    rt.table_drop(t.t.args[0].data).run(t.conn)
