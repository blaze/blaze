from ..expr import *
from operator import and_
from toolz import first, reduce
import inspect

def inject(t, ns=None):
    if not ns:
        ns = inspect.currentframe().f_back.f_locals
    for c in t.columns:
        ns[c] = t[c]


def Filter(t, *conditions):
    return t[reduce(and_, conditions)]


def desc(col):
    return -col


def arrange(t, *columns):
    return t.sort(list(columns))


def select(t, *columns):
    return t[[c.name for c in columns]]


def mutate(t, **kwargs):
    args = [t] + [v.label(k) for k, v in kwargs.items()]
    return merge(*args)


transform = mutate


class GroupBy(Expr):
    def __init__(self, child, *grouper):
        self.child = child
        if len(grouper) == 1:
            grouper = grouper[0]
        else:
            grouper = merge(*grouper)
        self.grouper = grouper

group_by = GroupBy


@dispatch(TableExpr)
def summarize(t, **kwargs):
    return summary(**kwargs)


n_distinct = nunique


n = count


@dispatch(GroupBy)
def summarize(t, **kwargs):
    return by(t.child, t.grouper, summary(**kwargs))
