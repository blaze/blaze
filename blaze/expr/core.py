from __future__ import absolute_import, division, print_function

import numbers
import toolz
import inspect
import functools
import datashape
from datashape import dshape, Record, DataShape

from toolz import unique, concat, compose, memoize, partial
import toolz
from pprint import pprint
from blaze.compatibility import StringIO, _strtypes, builtins

from .method_dispatch import select_functions
from ..dispatch import dispatch

__all__ = ['Expr', 'discover', 'path', 'common_subexpression']


def get_callable_name(o):
    """Welcome to str inception. Leave your kittens at home.
    """
    # special case partial objects
    if isinstance(o, functools.partial):
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
    else:
        stream = StringIO()
        pprint(s, stream=stream)
        return stream.getvalue().rstrip()


class Expr(object):
    __inputs__ = 'child',

    def __init__(self, *args, **kwargs):
        assert frozenset(kwargs).issubset(self.__slots__)

        for slot, arg in zip(self.__slots__, args):
            setattr(self, slot, arg)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __nonzero__(self): # pragma: no cover
        return True

    def __bool__(self):
        return True

    def get_field(self, fieldname):
        raise NotImplementedError()

    @property
    def args(self):
        return tuple(getattr(self, slot) for slot in self.__slots__)

    @property
    def inputs(self):
        return tuple(getattr(self, i) for i in self.__inputs__)

    @property
    def fields(self):
        if isinstance(self.dshape.measure, Record):
            return self.dshape.measure.names
        if hasattr(self, '_name'):
            return [self._name]

    def leaves(self):
        """ Leaves of an expresion tree

        All nodes without inputs.  Leaves are returned in order, left to right.

        >>> from blaze import TableSymbol, join, by

        >>> t = TableSymbol('t', '{id: int32, name: string}')
        >>> t.leaves()
        [t]
        >>> by(t.name, t.id.nunique()).leaves()
        [t]

        >>> v = TableSymbol('v', '{id: int32, city: string}')
        >>> join(t, v).leaves()
        [t, v]
        """

        if not self.inputs:
            return [self]
        else:
            return list(unique(concat(i.leaves() for i in self.inputs if
                                      isinstance(i, Expr))))

    def isidentical(self, other):
        return type(self) == type(other) and self.args == other.args

    __eq__ = isidentical

    def __hash__(self):
        return hash((type(self), self.args))

    def __str__(self):
        rep = ["%s=%s" % (slot, _str(arg))
               for slot, arg in zip(self.__slots__, self.args)]
        return "%s(%s)" % (type(self).__name__, ', '.join(rep))

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

    def subterms(self):
        return subterms(self)

    def __contains__(self, other):
        return other in set(self.subterms())

    def __getstate__(self):
        return self.args

    def __setstate__(self, state):
        self.__init__(*state)

    def __dir__(self):
        result = dir(type(self))
        if self.fields:
            result.extend(list(self.fields))

        d = dshape_methods(self.dshape)
        result.extend(list(d))

        return sorted(set(result))

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            if key[0] == '_':
                raise
            if self.fields and key in self.fields:
                return self.get_field(key)
            d = dshape_methods(self.dshape)
            if key in d:
                func = d[key]
                if func in method_properties:
                    return func(self)
                else:
                    return partial(func, self)
            else:
                raise AttributeError(key)

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



@dispatch(Expr)
def subterms(expr):
    return concat([[expr], concat(map(subterms, expr.inputs))])


@dispatch(numbers.Real)
def subterms(x):
    yield x


def subs(o, d):
    """ Substitute values within data structure

    >>> subs(1, {1: 2})
    2

    >>> subs([1, 2, 3], {2: 'Hello'})
    [1, 'Hello', 3]
    """
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


@dispatch(Expr, dict)
def _subs(o, d):
    """

    >>> from blaze.expr.table import TableSymbol
    >>> t = TableSymbol('t', '{name: string, balance: int}')
    >>> subs(t, {'balance': 'amount'}).fields
    ['name', 'amount']
    """
    newargs = [subs(arg, d) for arg in o.args]
    return type(o)(*newargs)


@dispatch(object, dict)
def _subs(o, d):
    """ Private dispatched version of ``subs``

    >>> subs('Hello', {})
    'Hello'
    """
    return o


@dispatch(Expr)
def discover(expr):
    return expr.dshape


def path(a, b):
    """ A path of nodes from a to b

    >>> from blaze.expr.table import TableSymbol
    >>> t = TableSymbol('t', '{name: string, amount: int, id: int}')
    >>> expr = t['amount'].sum()
    >>> list(path(expr, t))
    [sum(child=t['amount']), t['amount'], t]
    """
    while not a.isidentical(b):
        yield a
        if not a.inputs:
            break
        for child in a.inputs:
            if b in child.traverse():
                a = child
                break
    yield a


def common_subexpression(*exprs):
    """ Common sub expression between subexpressions

    Examples
    --------

    >>> from blaze import TableSymbol, common_subexpression

    >>> t = TableSymbol('t', '{x: int, y: int}')
    >>> common_subexpression(t['x'], t['y'])
    t
    """
    sets = [set(t.subterms()) for t in exprs]
    return builtins.max(set.intersection(*sets),
                        key=compose(len, str))

dshape_method_list = [
    ]

method_properties = set()

dshape_methods = memoize(partial(select_functions, dshape_method_list))
