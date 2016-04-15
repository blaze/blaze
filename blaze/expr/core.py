from __future__ import absolute_import, division, print_function

from collections import Mapping
import datetime
import numbers
import inspect

from pprint import pformat
from functools import reduce, partial

import toolz
from toolz import unique, concat, first

from ..compatibility import _strtypes
from ..dispatch import dispatch
from ..utils import ordered_intersect

__all__ = ['Node', 'path', 'common_subexpression', 'eval_str']


base = (numbers.Number,) + _strtypes + (datetime.datetime, datetime.timedelta)


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
    if a is b:
        return True
    if isinstance(a, base) and isinstance(b, base):
        return a == b
    if type(a) != type(b):
        return False
    if isinstance(a, Node):
        return all(map(isidentical, a._hashargs, b._hashargs))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(map(isidentical, a, b))
    return a == b


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
    __slots__ = ()
    __inputs__ = '_child',

    def __init__(self, *args, **kwargs):
        slots = set(self.__slots__)
        if not frozenset(slots) <= slots:
            raise TypeError('Unknown keywords: %s' % (set(kwargs) - slots))

        assigned = set()
        for slot, arg in zip(self.__slots__[1:], args):
            assigned.add(slot)
            setattr(self, slot, arg)

        for key, value in kwargs.items():
            if key in assigned:
                raise TypeError(
                    '%s got multiple values for argument %r' % (
                        type(self).__name__,
                        key,
                    ),
                )
            assigned.add(key)
            setattr(self, key, value)

        for slot in slots - assigned:
            setattr(self, slot, None)

    @property
    def _args(self):
        return tuple(getattr(self, slot) for slot in self.__slots__[1:])

    _hashargs = _args

    @property
    def _inputs(self):
        return tuple(getattr(self, i) for i in self.__inputs__)

    def _leaves(self):
        """ Leaves of an expression tree

        All nodes without inputs.  Leaves are returned in order, left to right.

        >>> from blaze.expr import symbol, join, by

        >>> t = symbol('t', 'var * {id: int32, name: string}')
        >>> t._leaves()
        [<`t` symbol; dshape='var * {id: int32, name: string}'>]
        >>> by(t.name, count=t.id.nunique())._leaves()
        [<`t` symbol; dshape='var * {id: int32, name: string}'>]

        >>> v = symbol('v', 'var * {id: int32, city: string}')
        >>> join(t, v)._leaves() == [t, v]
        True
        """

        if not self._inputs:
            return [self]
        else:
            return list(unique(concat(i._leaves() for i in self._inputs if
                                      isinstance(i, Node))))

    isidentical = isidentical

    def __hash__(self):
        hash_ = self._hash
        if hash_ is None:
            hash_ = self._hash = hash((type(self), self._hashargs))
        return hash_

    def __str__(self):
        rep = [
            '%s=%s' % (slot, _str(arg))
            for slot, arg in zip(self.__slots__[1:], self._args)
        ]
        return '%s(%s)' % (type(self).__name__, ', '.join(rep))

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
        return tuple(self._args)

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


def get_callable_name(o):
    """Welcome to str inception. Leave your kittens at home.
    """
    # special case partial objects
    if isinstance(o, partial):
        keywords = o.keywords
        kwds = (
            ', '.join('%s=%r' % item for item in keywords.items())
            if keywords else
            ''
        )
        args = ', '.join(map(repr, o.args))
        arguments = []
        if args:
            arguments.append(args)
        if kwds:
            arguments.append(kwds)
        return 'partial(%s, %s)' % (
            get_callable_name(o.func),
            ', '.join(arguments),
        )

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
        return repr(s)
    elif callable(s):
        return get_callable_name(s)
    elif isinstance(s, Node):
        return str(s)
    elif isinstance(s, (list, tuple)):
        body = ", ".join(_str(x) for x in s)
        return "({0})".format(body if len(s) > 1 else (body + ","))
    else:
        return pformat(s).rstrip()


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


@dispatch((tuple, list), Mapping)
def _subs(o, d):
    return type(o)([subs(arg, d) for arg in o])


@dispatch(Node, Mapping)
def _subs(o, d):
    """

    >>> from blaze.expr import symbol
    >>> t = symbol('t', 'var * {name: string, balance: int}')
    >>> subs(t, {'balance': 'amount'}).fields
    ['name', 'amount']
    """
    newargs = [subs(arg, d) for arg in o._args]
    return type(o)(*newargs)


@dispatch(object, Mapping)
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
    [sum(t.amount), t.amount, <`t` symbol; dshape='...'>]
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


def common_subexpression(expr, *exprs):
    """ Common sub expression between subexpressions

    Examples
    --------

    >>> from blaze.expr import symbol
    >>> t = symbol('t', 'var * {x: int, y: int}')
    >>> common_subexpression(t.x, t.y)
    <`t` symbol; dshape='var * {x: int32, y: int32}'>
    """
    # only one expression has itself as a common subexpression
    if not exprs:
        return expr

    exprs = (expr,) + exprs

    # get leaves for every expression
    all_leaves = [expr._leaves() for expr in exprs]

    # leaves common to all expressions
    leaves = set.intersection(*map(set, all_leaves))

    # no common leaves therefore no common subexpression
    if not leaves:
        raise ValueError(
            'No common leaves found in expressions %s' % list(exprs)
        )

    # list of paths from each expr to each leaf
    pathlist = [list(path(expr, leaf)) for expr in exprs for leaf in leaves]

    # ordered intersection of paths
    common = reduce(ordered_intersect, pathlist)
    if not common:
        raise ValueError(
            'No common subexpression found in paths to leaf: %s' % list(
                map(set, pathlist)
            )
        )

    # the first expression is the deepest node in the tree that is an ancestor
    # of every expression in `exprs`
    return first(common)


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
