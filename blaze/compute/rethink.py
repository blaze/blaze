import numbers

from ..expr import TableSymbol, Sort, Head, Distinct, Expr, Projection
from ..expr import Selection, Relational, Column
from ..expr import count, sum, min, max, mean, ScalarSymbol

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
    import ipdb; ipdb.set_trace()
    return t.with_fields(*p.columns)


@dispatch(Column, Table)
def compute_one(c, t):
    import ipdb; ipdb.set_trace()
    return t.with_fields(c.column)


@dispatch(Head, Table)
def compute_one(h, t):
    import ipdb; ipdb.set_trace()
    child = compute_one(h.child, t)
    return child.limit(h.n)


@dispatch(Sort, Table)
def compute_one(s, t):
    sort_order = {True: rt.asc, False: rt.desc}[s.ascending]
    return t.order_by(list(map(sort_order, s.key)))


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
