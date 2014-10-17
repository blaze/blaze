from datashape import *
import itertools

from .expressions import *
from .arithmetic import maxshape

__all__ = ['broadcast', 'Broadcast', 'scalar_symbols']

def broadcast(expr, leaves, scalars=None):
    scalars = scalars or scalar_symbols(leaves)
    assert len(scalars) == len(leaves)
    return Broadcast(leaves, scalars, expr._subs(dict(zip(leaves, scalars))))


class Broadcast(Expr):
    __slots__ = '_children', '_scalars', '_scalar_expr'

    @property
    def dshape(self):
        myshape = maxshape(map(shape, self._children))
        return DataShape(*(myshape + (self._scalar_expr.schema,)))


def scalar_symbols(exprs):
    """
    Gives a sequence of scalar symbols to mirror these expressions

    Examples
    --------

    >>> x = Symbol('x', '5 * 3 * int32')
    >>> y = Symbol('y', '5 * 3 * int32')

    >>> xx, yy = scalar_symbols([x, y])

    >>> xx._name, xx.dshape
    ('x', dshape("int32"))
    >>> yy._name, yy.dshape
    ('y', dshape("int32"))
    """
    new_names = ('_%d' % i for i in itertools.count(1))

    scalars = []
    names = set()
    for expr in exprs:
        if expr._name and expr._name not in names:
            name = expr._name
            names.add(name)
        else:
            name = next(new_names)

        s = Symbol(name, expr.schema)
        scalars.append(s)
    return scalars
