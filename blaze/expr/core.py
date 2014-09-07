from __future__ import absolute_import, division, print_function

import ast
import numbers
import toolz
import inspect
import functools
import operator as op
import pandas as pd
from toolz import unique, concat, merge
from pprint import pprint
import datashape as ds
import blaze as bz
from blaze.compatibility import StringIO, map
import math

from ..dispatch import dispatch

__all__ = ['Expr', 'discover', 'Lambda']


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

    @property
    def args(self):
        return tuple(getattr(self, slot) for slot in self.__slots__)

    @property
    def inputs(self):
        return tuple(getattr(self, i) for i in self.__inputs__)

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


@dispatch(Expr)
def subterms(expr):
    yield expr
    for i in expr.inputs:
        for node in subterms(i):
            yield node


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
    >>> subs(t, {'balance': 'amount'}).columns
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
        a = a.child
    yield a


class Expressify(ast.NodeVisitor):

    def __init__(self, scope):
        self.scope = scope

    def visit(self, node):
        result = super(Expressify, self).visit(node)

        if result is None:
            raise TypeError('%s nodes are not implemented' %
                            type(node).__name__)
        return result

    def visit_Num(self, node):
        return node.n

    def visit_Str(self, node):
        s = node.s

        # dateutil accepts the empty string as a valid datetime, don't let it do
        # that
        if s:
            try:
                return pd.Timestamp(s).to_pydatetime()
            except ValueError:
                return s
        return s

    def visit_Add(self, node):
        return op.add

    def visit_Sub(self, node):
        return op.sub

    def visit_Mult(self, node):
        return op.mul

    def visit_Div(self, node):
        return op.truediv

    def visit_Mod(self, node):
        return op.mod

    def visit_Pow(self, node):
        return op.pow

    def visit_Lt(self, node):
        return op.lt

    def visit_Gt(self, node):
        return op.gt

    def visit_Le(self, node):
        return op.le

    def visit_Ge(self, node):
        return op.ge

    def visit_Eq(self, node):
        return op.eq

    def visit_NotEq(self, node):
        return op.ne

    def visit_BitAnd(self, node):
        return op.and_

    def visit_BitOr(self, node):
        return op.or_

    def visit_Invert(self, node):
        return op.not_

    def visit_USub(self, node):
        return op.neg

    def visit_UnaryOp(self, node):
        f = self.visit(node.op)
        return f(self.visit(node.operand))

    def visit_Call(self, node):
        f = self.visit(node.func)
        return f(*map(self.visit, node.args))

    def visit_Attribute(self, node):
        return getattr(self.visit(node.value), node.attr)

    def visit_Compare(self, node):
        f = self.visit(node.ops[0])
        return f(self.visit(node.left), self.visit(node.comparators[0]))

    def visit_BinOp(self, node):
        f = self.visit(node.op)
        return f(self.visit(node.left), self.visit(node.right))

    def visit_Name(self, node):
        return self.scope[node.id]


class Lambda(Expr):

    __slots__ = 'child', 'expr', '_ast'

    __default_scope__ = toolz.keyfilter(lambda x: not x.startswith('__'),
                                        merge(math.__dict__))

    def __init__(self, child, expr, _ast=None):
        super(Lambda, self).__init__(child, expr, _ast=_ast)
        self._ast = _ast or ast.parse(str(expr), mode='eval').body

    @property
    def columns(self):
        return list(map(str, self.child.columns))

    @property
    def dshape(self):
        restype = self.expr.dshape
        argtypes = tuple(map(ds.dshape, self.child.schema[0].types))
        return ds.DataShape(ds.Function(*(argtypes + (restype,))))

    def __repr__(self):
        return 'lambda (%s): %s' % (', '.join(self.columns), self.expr)

    def __call__(self, row):
        scope = toolz.merge(self.__default_scope__,
                            dict(zip(self.columns, row)))
        parser = Expressify(scope)
        return parser.visit(self._ast)
