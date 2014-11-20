from __future__ import absolute_import, division, print_function

import datashape
from .expressions import Expr, schema_method_list

__all__ = ['Like', 'like']

class Like(Expr):
    """ Filter expression by string comparison

    >>> from blaze import symbol, like, compute
    >>> t = symbol('t', 'var * {name: string, city: string}')
    >>> expr = like(t, name='Alice*')

    >>> data = [('Alice Smith', 'New York'),
    ...         ('Bob Jones', 'Chicago'),
    ...         ('Alice Walker', 'LA')]
    >>> list(compute(expr, data))
    [('Alice Smith', 'New York'), ('Alice Walker', 'LA')]
    """
    __slots__ = '_hash', '_child', '_patterns'

    @property
    def patterns(self):
        return dict(self._patterns)

    @property
    def dshape(self):
        return datashape.var * self._child.dshape.subshape[0]


def like(child, **kwargs):
    return Like(child, tuple(kwargs.items()))
like.__doc__ = Like.__doc__

schema_method_list.extend([
    (lambda ds: 'string' in str(ds), set([like])),
    ])
