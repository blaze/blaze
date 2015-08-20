from __future__ import absolute_import, division, print_function

import datashape
from datashape import String
from datashape.predicates import isrecord
from odo.utils import copydoc

from .expressions import Expr, schema_method_list, ElemWise
from .arithmetic import Interp, Repeat, _mkbin, repeat, interp, _add, _radd

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


@copydoc(Like)
def like(child, **kwargs):
    return Like(child, tuple(kwargs.items()))


class UnaryStringFunction(ElemWise):

    """String function that only takes a single argument.
    """
    __slots__ = '_hash', '_child'


class strlen(UnaryStringFunction):
    schema = datashape.int64


def isstring(ds):
    measure = ds.measure
    return isinstance(getattr(measure, 'ty', measure), String)


_mod, _rmod = _mkbin('mod', Interp)
_mul, _rmul = _mkbin('mul', Repeat)


schema_method_list.extend([
    (isstring, set([_add, _radd, _mod, _rmod, _mul, _rmul, repeat, interp])),
    (lambda ds: isstring(ds) or (isrecord(ds.measure) and
                                 any(isinstance(getattr(typ, 'ty', typ),
                                                String)
                                     for typ in ds.measure.types)),
     set([like])),
    (lambda ds: isinstance(getattr(ds.measure, 'ty', ds.measure), String),
     set([strlen]))
])
