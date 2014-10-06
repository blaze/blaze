from __future__ import absolute_import, division, print_function

import datashape
from .expressions import Expr, schema_method_list

__all__ = ['Like', 'like']

class Like(Expr):
    __slots__ = '_child', '_patterns'

    @property
    def patterns(self):
        return dict(self._patterns)

    @property
    def dshape(self):
        return datashape.var * self._child.dshape.subshape[0]


def like(child, **kwargs):
    """ Filter expression by string comparison

    >>> from blaze import TableSymbol, like, compute
    >>> t = TableSymbol('t', '{name: string, city: string}')
    >>> expr = like(t, name='Alice*')

    >>> data = [('Alice Smith', 'New York'),
    ...         ('Bob Jones', 'Chicago'),
    ...         ('Alice Walker', 'LA')]
    >>> list(compute(expr, data))
    [('Alice Smith', 'New York'), ('Alice Walker', 'LA')]
    """
    return Like(child, tuple(kwargs.items()))

from .table import min, max

schema_method_list.extend([
    (lambda ds: 'string' in str(ds), set([like, min, max])),
    ])
