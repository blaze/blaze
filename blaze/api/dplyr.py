from ..expr import *
from operator import and_
from toolz import first, reduce
import inspect

def inject(t, ns=None):
    """ Inject columns into local namespace

    >>> t = TableSymbol('t', '{x: int, y: int}')
    >>> inject(t)

    >>> x
    t['x']

    >>> x + y
    t['x'] + t['y']
    """
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
    """ Select columns from table

    >>> t = TableSymbol('t', '{x: int, y: int, z: int}')
    >>> select(t, t.x, t.z)
    t[['x', 'z']]
    """
    return t[[c.name for c in columns]]


def transform(t, replace=True, **kwargs):
    """ Add named columns to table

    >>> t = TableSymbol('t', '{x: int, y: int}')
    >>> transform(t, xy=t.x + t.y).columns
    ['x', 'y', 'xy']
    """
    if replace and set(t.columns).intersection(set(kwargs)):
        t = t[[c for c in t.columns if c not in kwargs]]

    args = [t] + [v.label(k) for k, v in kwargs.items()]
    return merge(*args)


mutate = transform


class GroupBy(Expr):
    """ A Group By object

    To be operated on by ``blaze.dplyr.api.summarize``

    >>> t = TableSymbol('t', '{x: int, y: int}')
    >>> g = group_by(t, t.x)
    >>> summarize(g, total=t.y.sum()).columns
    ['x', 'total']
    """
    __slots__ = ['grouper']

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
    return by(t.grouper, summary(**kwargs))
