from __future__ import absolute_import, division, print_function

from collections import Mapping, OrderedDict
import datetime
from functools import reduce, partial
import inspect
from itertools import repeat
import numbers

from pprint import pformat
from weakref import WeakValueDictionary

import toolz
from toolz import unique, concat, first

from ..compatibility import _strtypes
from ..dispatch import dispatch
from ..utils import ordered_intersect

__all__ = ['Node', 'path', 'common_subexpression', 'eval_str']


base = (numbers.Number,) + _strtypes + (datetime.datetime, datetime.timedelta)


def _resolve_args(cls, *args, **kwargs):
    """Resolve the arguments from a node class into an ordereddict.
    All arguments are assumed to have a default of None.

    This is sort of like getargspec but uses the `Node` specific machinery.

    Parameters
    ----------
    cls : subclass of Node
        The class to resolve the arguments for.
    *args, **kwargs
        The arguments that were passed.

    Returns
    -------
    args : OrderedDict
        A dictionary mapping argument names to their value in the order
        they appear in the `_arguments` tuple.

    Examples
    --------
    >>> class MyNode(Node):
    ...     _arguments = 'a', 'b', 'c'
    ...

    good cases
    >>> _resolve_args(MyNode, 1, 2, 3)
    OrderedDict([('a', 1), ('b', 2), ('c', 3)])
    >>> _resolve_args(MyNode, 1, 2, c=3)
    OrderedDict([('a', 1), ('b', 2), ('c', 3)])
    >>> _resolve_args(MyNode, a=1, b=2, c=3)
    OrderedDict([('a', 1), ('b', 2), ('c', 3)])

    error cases
    >>> _resolve_args(MyNode, 1, 2, 3, a=4)
    Traceback (most recent call last):
       ...
    TypeError: MyNode got multiple values for argument 'a'
    >>> _resolve_args(MyNode, 1, 2, 3, 4)
    Traceback (most recent call last):
       ...
    TypeError: MyNode takes 3 positional arguments but 4 were given
    >>> _resolve_args(MyNode, 1, 2, 3, d=4)
    Traceback (most recent call last):
       ...
    TypeError: MyNode got unknown keywords: d
    """
    attrs = cls._arguments
    attrset = set(attrs)
    if not set(kwargs) <= attrset:
        raise TypeError(
            '%s got unknown keywords: %s' % (
                cls.__name__,
                ', '.join(set(kwargs) - attrset),
            ),
        )

    if len(args) > len(attrs):
        raise TypeError(
            '%s takes 3 positional arguments but %d were given' % (
                cls.__name__,
                len(args),
            ),
        )

    attributes = OrderedDict(zip(attrs, repeat(None)))
    to_add = dict(zip(attrs, args))
    attributes.update(to_add)
    added = set(to_add)

    for key, value in kwargs.items():
        if key in added:
            raise TypeError(
                '%s got multiple values for argument %r' % (
                    cls.__name__,
                    key,
                ),
            )
        attributes[key] = value
        added.add(key)

    return attributes


def _static_identity(ob):
    return type(ob)._static_identity(*ob._args)


def _setattr(ob, name, value):
    object.__setattr__(ob, name, value)
    return value


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
    _arguments = '_child',
    __inputs__ = '_child',
    __expr_instance_cache = WeakValueDictionary()

    def __new__(cls, *args, **kwargs):
        static_id = cls._static_identity(*args, **kwargs)
        try:
            return cls.__expr_instance_cache[static_id]
        except KeyError:
            cls.__expr_instance_cache[static_id] = self = super(
                Node,
                cls,
            ).__new__(cls)._init(*args, **kwargs)
            return self

    def _init(self, *args, **kwargs):
        for name, arg in _resolve_args(type(self), *args, **kwargs).items():
            _setattr(self, name, arg)

        _setattr(self, '_hash', None)
        return self

    def __setattr__(self, name, value):
        raise AttributeError('cannot set attributes of immutable objects')

    @property
    def _args(self):
        return tuple(getattr(self, slot) for slot in self._arguments)

    @classmethod
    def _static_identity(cls, *args, **kwargs):
        return (cls,) + tuple(_resolve_args(cls, *args, **kwargs).values())

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

    def isidentical(self, other):
        """Identity check for blaze expressions.
        """
        return self is other

    def __hash__(self):
        hash_ = self._hash
        if hash_ is None:
            hash_ = _setattr(
                self,
                '_hash',
                hash((type(self), _static_identity(self))),
            )
        return hash_

    def __str__(self):
        rep = [
            '%s=%s' % (slot, _str(arg))
            for slot, arg in zip(self._arguments, self._args)
        ]
        return '%s(%s)' % (type(self).__name__, ', '.join(rep))

    def _traverse(self):
        """ Traverse over tree, yielding all subtrees and leaves """
        yield self
        traversals = (
            arg._traverse() if isinstance(arg, Node) else [arg]
            for arg in self._args
        )
        for item in concat(traversals):
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

    def __reduce_ex__(self, protocol):
        if protocol < 2:
            raise ValueError(
                'blaze expressions may only be pickled with protocol'
                ' 2 or greater',
            )
        return type(self), self._args

    def __eq__(self, other):
        try:
            return self.isidentical(other) or self._eq(other)
        except AttributeError:
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
    d = {k: v for k, v in d.items() if k is not v}
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
    return type(o)(subs(arg, d) for arg in o)


@dispatch(Node, Mapping)
def _subs(o, d):
    """

    >>> from blaze.expr import symbol
    >>> t = symbol('t', 'var * {name: string, balance: int}')
    >>> subs(t, {'balance': 'amount'}).fields
    ['name', 'amount']
    """
    newargs = (subs(arg, d) for arg in o._args)
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
