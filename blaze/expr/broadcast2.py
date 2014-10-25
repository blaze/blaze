from datashape import *
from datashape.predicates import iscollection
import itertools
from toolz import curry

from .expressions import *
from .expressions import Field, Map
from .arithmetic import maxshape, Arithmetic, Add
from .math import Math
from .datetime import DateTime

__all__ = ['broadcast', 'Broadcast', 'scalar_symbols', 'broadcast_collect']

def broadcast(expr, leaves, scalars=None):
    scalars = scalars or scalar_symbols(leaves)
    assert len(scalars) == len(leaves)
    return Broadcast(tuple(leaves), tuple(scalars), expr._subs(dict(zip(leaves, scalars))))


class Broadcast(Expr):
    __slots__ = '_children', '_scalars', '_scalar_expr'

    @property
    def dshape(self):
        myshape = maxshape(map(shape, self._children))
        return DataShape(*(myshape + (self._scalar_expr.schema,)))

    @property
    def _inputs(self):
        return self._children

    @property
    def _name(self):
        return self._scalar_expr._name


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


def broadcast_collect(broadcastable_types, expr):
    """ Collapse expression down using Broadcast

    Expressions of type Broadcastable_types are swallowed into Broadcast
    operations

    >>> x = Symbol('x', '5 * 3 * int32')
    >>> y = Symbol('y', '5 * 3 * int32')

    >>> expr = (x + 2*y)

    >>> broadcast_collect((Field, Arithmetic), x + 2*y)
    Broadcast(_children=(x, y), _scalars=(x, y), _scalar_expr=x + (2 * y))

    >>> broadcast_collect((Field, Add), x + 2*y)
    Broadcast(_children=(2 * y, x), _scalars=(y, x), _scalar_expr=x + y)
    """
    if isinstance(expr, broadcastable_types) and iscollection(expr.dshape):
        leaves = leaves_of_type(broadcastable_types, expr)
        expr = broadcast(expr, sorted(leaves, key=str))

    children = [broadcast_collect(broadcastable_types, child)
                for child in expr._inputs]
    return expr._subs(dict(zip(expr._inputs, children)))

@curry
def leaves_of_type(types, expr):
    """ Leaves of an expression skipping all operations of type ``types``
    """
    if not isinstance(expr, types):
        return set([expr])
    else:
        return set.union(*map(leaves_of_type(types), expr._inputs))


TableBroadcastable = (Field, Arithmetic, Map, DateTime, Math)
def broadcast_table_collect(expr):
    if isinstance(expr, Symbol):
        return expr
    if not isinstance(expr, TableBroadcastable) or ndim(expr) == 0:
        children = map(broadcast_table_collect, expr._inputs)
        return expr._subs(dict(zip(expr._inputs, children)))

    leaves = _table_find_leaves(expr)
    return broadcast(expr, leaves)


def _table_find_leaves(expr):
    """ Find extent of broadcast table optimization

    helper to broadcast_table_collect
    """
    if not isinstance(expr, TableBroadcastable):
        return set([expr])
    leaves = set()
    for child in expr._inputs:
        if isinstance(child, TableBroadcastable) and child.shape == expr.shape:
            leaves |= _table_find_leaves(child)
        else:
            leaves.add(child)
    return leaves


