from __future__ import absolute_import, division, print_function

import datashape
from datashape import String
from datashape.predicates import isrecord, iscollection
from .expressions import Expr, schema_method_list, ElemWise

__all__ = ['Like', 'like', 'strlen', 'UnaryStringFunction']


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


class UnaryStringFunction(ElemWise):

    """String function that only takes a single argument.
    """
    __slots__ = '_hash', '_child'


class strlen(UnaryStringFunction):
    schema = datashape.int64


def isstring(ds):
    measure = ds.measure
    return isinstance(getattr(measure, 'ty', measure), String)


schema_method_list.extend([
    (lambda ds: isstring(ds) or (isrecord(ds.measure) and
                                 any(isinstance(getattr(typ, 'ty', typ),
                                                String)
                                     for typ in ds.measure.types)),
     set([like])),
    (lambda ds: isinstance(getattr(ds.measure, 'ty', ds.measure), String),
     set([strlen]))
])
