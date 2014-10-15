from ..expr import *
from ..dispatch import dispatch
from operator import and_
from toolz import first, reduce
import inspect

__all__ = ['inject', 'Filter', 'desc', 'select', 'arrange', 'transform',
        'group_by', 'summarize', 'n', 'mutate', 'n_distinct']


def inject(t, ns=None):
    """ Inject columns into local namespace

    >>> t = Symbol('t', 'var * {x: int, y: int}')
    >>> inject(t)

    >>> x
    t['x']

    >>> x + y
    t['x'] + t['y']
    """
    if not ns:
        ns = inspect.currentframe().f_back.f_locals
    for c in t.fields:
        ns[c] = t[c]


def Filter(t, *conditions):
    return t[reduce(and_, conditions)]


def desc(col):
    return -col


def arrange(t, *columns):
    return t.sort(list(columns))


def select(t, *columns):
    """ Select columns from table

    >>> t = Symbol('t', 'var * {x: int, y: int, z: int}')
    >>> select(t, t.x, t.z)
    t[['x', 'z']]
    """
    return t[[c._name for c in columns]]


def transform(t, replace=True, **kwargs):
    """ Add named columns to table

    >>> t = Symbol('t', 'var * {x: int, y: int}')
    >>> transform(t, xy=t.x + t.y).fields
    ['x', 'y', 'xy']
    """
    if replace and set(t.fields).intersection(set(kwargs)):
        t = t[[c for c in t.fields if c not in kwargs]]

    args = [t] + [v.label(k) for k, v in kwargs.items()]
    return merge(*args)


mutate = transform


class GroupBy(Expr):
    """ A Group By object

    To be operated on by ``blaze.dplyr.api.summarize``

    >>> t = Symbol('t', 'var * {x: int, y: int}')
    >>> g = group_by(t, t.x)
    >>> summarize(g, total=t.y.sum()).fields
    ['x', 'total']
    """
    __slots__ = ['_child', 'grouper']

    def __init__(self, child, *grouper):
        self._child = child
        if len(grouper) == 1:
            grouper = grouper[0]
        else:
            grouper = merge(*grouper)
        self.grouper = grouper

group_by = GroupBy


@dispatch(Expr)
def summarize(t, **kwargs):
    return summary(**kwargs)


n_distinct = nunique


n = count


@dispatch(GroupBy)
def summarize(t, **kwargs):
    return by(t.grouper, summary(**kwargs))
