import numbers

from ..expr import TableSymbol, Sort, Head, Distinct, Expr, Projection
from ..expr import Selection, Relational
from ..expr import count, sum, min, max, mean, ScalarSymbol

from ..compatibility import basestring

from ..dispatch import dispatch

import rethinkdb as rt
from rethinkdb.ast import Table
from rethinkdb.net import Connection


__all__ = ['compute_one']


@dispatch(TableSymbol, Table)
def compute_one(_, t):
    return t


@dispatch(Projection, Table)
def compute_one(p, t):
    return compute_one(p.child, t).pluck(*p.columns)


@dispatch(Head, Table)
def compute_one(h, t):
    return compute_one(h.child, t).limit(h.n)


@dispatch(basestring)
def default_sort_order(key, f=rt.asc):
    return f(key),


@dispatch(list)
def default_sort_order(keys, f=rt.asc):
    for key in keys:
        yield f(key)


@dispatch(Sort, Table)
def compute_one(s, t):
    f = rt.asc if s.ascending else rt.desc
    child_result = compute_one(s.child, t)
    return child_result.order_by(*default_sort_order(s.key, f=f))


@dispatch(count, Table)
def compute_one(c, t):
    return t.count(c.dshape[0].names[0])


@dispatch((count, sum, min, max, mean), Table)
def compute_one(f, t):
    return getattr(t, type(f).__name__)(f.dshape[0].names[0])


@dispatch(ScalarSymbol, Table)
def compute_one(ss, _):
    return rt.row[ss.name]


@dispatch((basestring, numbers.Real), Table)
def compute_one(s, _):
    return s


@dispatch(Relational, Table)
def compute_one(r, t):
    return r.op(compute_one(r.lhs, t), compute_one(r.rhs, t))


@dispatch(Selection, Table)
def compute_one(s, t):
    e = s.predicate.expr
    return t.filter(e.op(compute_one(e.lhs, t), compute_one(e.rhs, t)))


@dispatch(Distinct, Table)
def compute_one(d, t):
    return t.with_fields(*d.columns).distinct()


@dispatch(Expr, Table, Connection)
def compute_one(e, t, c):
    return list(compute_one(e, t).run(c))
