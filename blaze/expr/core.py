from __future__ import absolute_import, division, print_function

import toolz

from ..dispatch import dispatch

__all__ = ['Expr', 'Scalar', 'discover']


def _str(s):
    """ Wrap single quotes around strings """
    if isinstance(s, str):
        return "'%s'" % s
    else:
        return str(s)


class Expr(object):
    __inputs__ = 'parent',
    @property
    def args(self):
        return tuple(getattr(self, slot) for slot in self.__slots__)

    @property
    def inputs(self):
        return tuple(getattr(self, i) for i in self.__inputs__)

    def isidentical(self, other):
        return type(self) == type(other) and self.args == other.args

    __eq__ = isidentical

    def __hash__(self):
        return hash((type(self), self.args))

    def __str__(self):
        return "%s(%s)" % (type(self).__name__, ', '.join(map(_str, self.args)))

    def __repr__(self):
        return str(self)

    def traverse(self):
        """ Traverse over tree, yielding all subtrees and leaves """
        yield self
        traversals = (arg.traverse() if isinstance(arg, Expr) else [arg]
                        for arg in self.args)
        for trav in traversals:
            for item in trav:
                yield item

    def subs(self, d):
        """ Substitute terms in the tree

        >>> from blaze.expr.table import TableSymbol
        >>> t = TableSymbol('t', '{name: string, amount: int, id: int}')
        >>> expr = t['amount'] + 3
        >>> expr.subs({3: 4, 'amount': 'id'}).isidentical(t['id'] + 4)
        True
        """
        return subs(self, d)

    def resources(self):
        return toolz.merge([arg.resources() for arg in self.args
                                            if isinstance(arg, Expr)])

    def ancestors(self):
        yield self
        for i in self.inputs:
            for node in i.ancestors():
                yield node

    def __contains__(self, other):
        return other in set(self.ancestors())


def subs(o, d):
    if o in d:
        d = d.copy()
        other = d.pop(o)
        return subs(other, d)
    if isinstance(o, (tuple, list)):
        return type(o)([subs(arg, d) for arg in o])
    if hasattr(o, 'args'):
        newargs = [subs(arg, d) for arg in o.args]
        return type(o)(*newargs)

    return o


class Scalar(Expr):
    pass


@dispatch(Expr)
def discover(expr):
    return expr.dshape


def path(a, b):
    """ A path of nodes from a to b

    >>> from blaze.expr.table import TableSymbol
    >>> t = TableSymbol('t', '{name: string, amount: int, id: int}')
    >>> expr = t['amount'].sum()
    >>> list(path(expr, t))
    [sum(t['amount']), t['amount'], t]
    """
    while not a.isidentical(b):
        yield a
        a = a.parent
    yield a
