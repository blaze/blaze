import numbers

from ..expr import TableSymbol, Sort, Head, Distinct, Expr, Projection
from ..expr import Selection, Relational, ScalarSymbol
from ..expr import count, sum, min, max, mean, nunique

from ..compatibility import basestring

from ..dispatch import dispatch

import rethinkdb as rt
from rethinkdb.ast import Table as RTable
from rethinkdb.net import Connection


__all__ = ['compute_one']


@dispatch(TableSymbol, RTable)
def compute_one(_, t):
    return t


@dispatch(Projection, RTable)
def compute_one(p, t):
    return compute_one(p.child, t).pluck(*p.columns)


@dispatch(Head, RTable)
def compute_one(h, t):
    return compute_one(h.child, t).limit(h.n)


@dispatch(basestring)
def default_sort_order(key, f=rt.asc):
    return f(key),


@dispatch(list)
def default_sort_order(keys, f=rt.asc):
    for key in keys:
        yield f(key)


@dispatch(Sort, RTable)
def compute_one(s, t):
    f = rt.asc if s.ascending else rt.desc
    child_result = compute_one(s.child, t)
    return child_result.order_by(*default_sort_order(s.key, f=f))


@dispatch(sum, RTable)
def compute_one(f, t):
    return compute_one(f.child, t).sum(f.child.column)


@dispatch((min, max), RTable)
def compute_one(f, t):
    return getattr(compute_one(f.child, t),
                   type(f).__name__)(f.child.column)[f.child.column]


@dispatch(count, RTable)
def compute_one(f, t):
    return compute_one(f.child, t).count()


@dispatch(mean, RTable)
def compute_one(f, t):
    return compute_one(f.child, t).avg(f.child.column)


@dispatch(nunique, RTable)
def compute_one(f, t):
    return compute_one(f.child, t).distinct().count()


@dispatch(ScalarSymbol, RTable)
def compute_one(ss, _):
    return rt.row[ss.name]


@dispatch((basestring, numbers.Real), RTable)
def compute_one(s, _):
    return s


@dispatch(Relational, RTable)
def compute_one(r, t):
    return r.op(compute_one(r.lhs, t), compute_one(r.rhs, t))


@dispatch(Selection, RTable)
def compute_one(s, t):
    e = s.predicate.expr
    return t.filter(e.op(compute_one(e.lhs, t), compute_one(e.rhs, t)))


@dispatch(Distinct, RTable)
def compute_one(d, t):
    return t.with_fields(*d.columns).distinct()


@dispatch(Expr, RTable, Connection)
def compute_one(e, t, c):
    result = compute_one(e, t).run(c)
    try:
        return list(result)
    except TypeError:
        return result
