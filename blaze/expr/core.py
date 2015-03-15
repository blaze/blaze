from __future__ import absolute_import, division, print_function

import numbers
import toolz
import inspect

from toolz import unique, concat, compose, partial
import toolz
from pprint import pprint

from ..compatibility import StringIO, _strtypes, builtins
from ..dispatch import dispatch

__all__ = ['Node', 'path', 'common_subexpression', 'eval_str']


base = (numbers.Number,) + _strtypes

class Node(object):
    """ Node in a tree

    This serves as the base class for ``Expr``.  This class holds all of the
    tree traversal functions that are independent of tabular or array
    computation.  This is everything that we can do independent of the problem
    domain.  Note that datashape is not imported.

    See Also
    --------

    blaze.expr.expressions.Expr
    """
    __inputs__ = '_child',

    def __init__(self, *args, **kwargs):
        assert frozenset(kwargs).issubset(self.__slots__)

        for slot, arg in zip(self.__slots__[1:], args):
            setattr(self, slot, arg)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def _args(self):
        return tuple([getattr(self, slot) for slot in self.__slots__[1:]])

    @property
    def _inputs(self):
        return tuple([getattr(self, i) for i in self.__inputs__])

    def _leaves(self):
        """ Leaves of an expression tree

        All nodes without inputs.  Leaves are returned in order, left to right.

        >>> from blaze.expr import symbol, join, by

        >>> t = symbol('t', 'var * {id: int32, name: string}')
        >>> t._leaves()
        [t]
        >>> by(t.name, count=t.id.nunique())._leaves()
        [t]

        >>> v = symbol('v', 'var * {id: int32, city: string}')
        >>> join(t, v)._leaves()
        [t, v]
        """

        if not self._inputs:
            return [self]
        else:
            return list(unique(concat(i._leaves() for i in self._inputs if
                                      isinstance(i, Node))))

    def isidentical(self, other):
        return isidentical(self, other)

    def __hash__(self):
        try:
            return self._hash
        except AttributeError:
            self._hash = hash((type(self), self._args))
            return self._hash

    def __str__(self):
        rep = ["%s=%s" % (slot, _str(arg))
                for slot, arg in zip(self.__slots__[1:], self._args)]
        return "%s(%s)" % (type(self).__name__, ', '.join(rep))

    def __repr__(self):
        return str(self)

    def _traverse(self):
        """ Traverse over tree, yielding all subtrees and leaves """
        yield self
        traversals = (arg._traverse() if isinstance(arg, Node) else [arg]
                      for arg in self._args)
        for trav in traversals:
            for item in trav:
                yield item

    def _subs(self, d):
        """ Substitute terms in the tree

        >>> from blaze.expr import symbol
        >>> t = symbol('t', 'var * {name: string, amount: int, id: int}')
        >>> expr = t.amount + 3
        >>> expr._subs({3: 4, 'amount': 'id'}).isidentical(t.id + 4)
        True
        """
        return subs(self, d)

    def _resources(self):
        return toolz.merge([arg._resources() for arg in self._args
                            if isinstance(arg, Node)])

    def _subterms(self):
        return subterms(self)

    def __contains__(self, other):
        return other in set(self._subterms())

    def __getstate__(self):
        return self._args

    def __setstate__(self, state):
        self.__init__(*state)

    def __eq__(self, other):
        ident = self.isidentical(other)
        if ident is True:
            return ident

        try:
            return self._eq(other)
        except AttributeError:
            # e.g., we can't compare whole tables to other things (yet?)
            pass
        return False

    def __ne__(self, other):
        return self._ne(other)

    def __lt__(self, other):
        return self._lt(other)

    def __le__(self, other):
        return self._le(other)

    def __gt__(self, other):
        return self._gt(other)

    def __ge__(self, other):
        return self._ge(other)

    def __add__(self, other):
        return self._add(other)

    def __radd__(self, other):
        return self._radd(other)

    def __mul__(self, other):
        return self._mul(other)

    def __rmul__(self, other):
        return self._rmul(other)

    def __div__(self, other):
        return self._div(other)

    def __rdiv__(self, other):
        return self._rdiv(other)

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    def __floordiv__(self, other):
        return self._floordiv(other)

    def __rfloordiv__(self, other):
        return self._rfloordiv(other)

    def __sub__(self, other):
        return self._sub(other)

    def __rsub__(self, other):
        return self._rsub(other)

    def __pow__(self, other):
        return self._pow(other)

    def __rpow__(self, other):
        return self._rpow(other)

    def __mod__(self, other):
        return self._mod(other)

    def __rmod__(self, other):
        return self._rmod(other)

    def __or__(self, other):
        return self._or(other)

    def __ror__(self, other):
        return self._ror(other)

    def __and__(self, other):
        return self._and(other)

    def __rand__(self, other):
        return self._rand(other)

    def __neg__(self):
        return self._neg()

    def __invert__(self):
        return self._invert()

    def __abs__(self):
        from .math import abs
        return abs(self)


def isidentical(a, b):
    """ Strict equality testing

    Different from x == y -> Eq(x, y)

    >>> isidentical(1, 1)
    True

    >>> from blaze.expr import symbol
    >>> x = symbol('x', 'int')
    >>> isidentical(x, 1)
    False

    >>> isidentical(x + 1, x + 1)
    True

    >>> isidentical(x + 1, x + 2)
    False

    >>> isidentical((x, x + 1), (x, x + 1))
    True

    >>> isidentical((x, x + 1), (x, x + 2))
    False
    """
    if isinstance(a, base) and isinstance(b, base):
        return a == b
    if type(a) != type(b):
        return False
    if isinstance(a, Node):
        return all(map(isidentical, a._args, b._args))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(map(isidentical, a, b))
    return a == b


def get_callable_name(o):
    """Welcome to str inception. Leave your kittens at home.
    """
    # special case partial objects
    if isinstance(o, partial):
        return 'partial(%s, %s)' % (get_callable_name(o.func),
                                    ', '.join(map(str, o.args)))

    try:
        # python 3 makes builtins look nice
        return o.__qualname__
    except AttributeError:
        try:
            # show the module of the object, if we can
            return '%s.%s' % (inspect.getmodule(o).__name__, o.__name__)
        except AttributeError:
            try:
                # __self__ tells us the class the method is bound to
                return '%s.%s' % (o.__self__.__name__, o.__name__)
            except AttributeError:
                # exhausted all avenues of printing callables so just print the
                # name of the object
                return o.__name__


def _str(s):
    """ Wrap single quotes around strings """
    if isinstance(s, str):
        return "'%s'" % s
    elif callable(s):
        return get_callable_name(s)
    elif isinstance(s, Node):
        return str(s)
    else:
        stream = StringIO()
        pprint(s, stream=stream)
        return stream.getvalue().rstrip()


@dispatch(Node)
def subterms(expr):
    return concat([[expr], concat(map(subterms, expr._inputs))])


@dispatch(object)
def subterms(x):
    yield x


def subs(o, d):
    """ Substitute values within data structure

    >>> subs(1, {1: 2})
    2

    >>> subs([1, 2, 3], {2: 'Hello'})
    [1, 'Hello', 3]
    """
    d = dict((k, v) for k, v in d.items() if k is not v)
    if not d:
        return o
    try:
        if o in d:
            d = d.copy()
            o = d.pop(o)
    except TypeError:
        pass
    return _subs(o, d)


@dispatch((tuple, list), dict)
def _subs(o, d):
    return type(o)([subs(arg, d) for arg in o])


@dispatch(Node, dict)
def _subs(o, d):
    """

    >>> from blaze.expr import symbol
    >>> t = symbol('t', 'var * {name: string, balance: int}')
    >>> subs(t, {'balance': 'amount'}).fields
    ['name', 'amount']
    """
    newargs = [subs(arg, d) for arg in o._args]
    return type(o)(*newargs)


@dispatch(object, dict)
def _subs(o, d):
    """ Private dispatched version of ``subs``

    >>> subs('Hello', {})
    'Hello'
    """
    return o


def path(a, b):
    """ A path of nodes from a to b

    >>> from blaze.expr import symbol
    >>> t = symbol('t', 'var * {name: string, amount: int, id: int}')
    >>> expr = t.amount.sum()
    >>> list(path(expr, t))
    [sum(t.amount), t.amount, t]
    """
    while not a.isidentical(b):
        yield a
        if not a._inputs:
            break
        for child in a._inputs:
            if any(b.isidentical(node) for node in child._traverse()):
                a = child
                break
    yield a


def common_subexpression(*exprs):
    """ Common sub expression between subexpressions

    Examples
    --------

    >>> from blaze.expr import symbol, common_subexpression

    >>> t = symbol('t', 'var * {x: int, y: int}')
    >>> common_subexpression(t.x, t.y)
    t
    """
    sets = [set(subterms(t)) for t in exprs]
    return builtins.max(set.intersection(*sets),
                        key=compose(len, str))


def eval_str(expr):
    """ String suitable for evaluation

    >>> from blaze.expr import symbol, eval_str
    >>> x = symbol('x', 'real')
    >>> eval_str(2*x + 1)
    '(2 * x) + 1'

    >>> from datetime import date
    >>> eval_str(date(2000, 1, 20))
    'datetime.date(2000, 1, 20)'
    """
    from datetime import date, datetime
    if isinstance(expr, (date, datetime)):
        return repr(expr)
    return repr(expr) if isinstance(expr, _strtypes) else str(expr)


def parenthesize(s):
    """

    >>> parenthesize('1')
    '1'
    >>> parenthesize('1 + 2')
    '(1 + 2)'
    """
    if ' ' in s:
        return '(%s)' % s
    else:
        return s
