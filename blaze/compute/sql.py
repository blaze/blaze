
from blaze.objects.table import *
from multipledispatch import dispatch
import sqlalchemy as sa
import sqlalchemy

base = (int, float, str, bool)

@dispatch(Projection, sqlalchemy.Table)
def compute(t, s):
    s = compute(t.table, s)
    return sa.select([s.c.get(col) for col in t.columns])


@dispatch(Column, sqlalchemy.Table)
def compute(t, s):
    s = compute(t.table, s)
    return s.c.get(t.columns[0])


@dispatch(base, object)
def compute(a, b):
    return a


@dispatch(Relational, sqlalchemy.Table)
def compute(t, s):
    return t.op(compute(t.lhs, s), compute(t.rhs, s))


@dispatch(Selection, sqlalchemy.Table)
def compute(t, s):
    return sa.select([compute(t.table, s)]).where(compute(t.predicate, s))


@dispatch(Table, sqlalchemy.Table)
def compute(t, s):
    return s
