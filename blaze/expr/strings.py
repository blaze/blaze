import datashape
from .expr import Collection

__all__ = ['Like', 'like']

class Like(Collection):
    __slots__ = 'child', '_patterns'

    @property
    def patterns(self):
        return dict(self._patterns)

    @property
    def dshape(self):
        return datashape.var * self.child.dshape.subshape[0]


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

from .expr import schema_method_list
from .table import min, max

schema_method_list.extend([
    (lambda ds: 'string' in str(ds), set([like, min, max])),
    ])
