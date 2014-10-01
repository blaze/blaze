from __future__ import absolute_import, division, print_function

import numbers
import toolz
import inspect
import functools
import datashape
from datashape import dshape, DataShape, Record

from toolz import unique, concat, memoize, partial
import toolz
from pprint import pprint
from blaze.compatibility import StringIO, _strtypes, builtins

from .method_dispatch import select_functions
from ..dispatch import dispatch

__all__ = ['Expr', 'Symbol', 'discover', 'path', 'ElemWise', 'Projection',
    'projection']


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

    def _len(self):
        try:
            return int(self.dshape[0])
        except TypeError:
            raise ValueError('Can not determine length of table with the '
                    'following datashape: %s' % self.dshape)

    def __len__(self): # pragma: no cover
        return self._len()

    def get_field(self, fieldname):
        return Field(self, fieldname)

    def __getitem__(self, key):
        if isinstance(key, str) and key in self.names:
            return self.get_field(key)
        if (isinstance(key, list)
                and builtins.all(isinstance(k, _strtypes) for k in key)):
            if set(key).issubset(self.names):
                return self.project(key)
            else:
                raise ValueError('Names %s not consistent with known names %s'
                        % (key, self.names))
        raise NotImplementedError("Not understood %s[%s]" % (self, key))

    def project(self, key):
        return projection(self, key)

    @property
    def args(self):
        return tuple(getattr(self, slot) for slot in self.__slots__)

    @property
    def inputs(self):
        return tuple(getattr(self, i) for i in self.__inputs__)

    @property
    def shape(self):
        s = list(self.dshape.shape)
        for i, elem in enumerate(s):
            try:
                s[i] = int(elem)
            except TypeError:
                pass

        return tuple(s)

    @property
    def schema(self):
        return datashape.dshape(self.dshape.measure)

    @property
    def dtype(self):
        ds = self.schema[-1]
        if isinstance(ds, Record):
            if len(ds.fields) > 1:
                raise TypeError("`.dtype` not defined for multicolumn object. "
                                "Use `.schema` instead")
            else:
                return dshape(first(ds.types))
        else:
            return dshape(ds)

    @property
    def names(self):
        if isinstance(self.dshape.measure, datashape.Record):
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
        if self.names:
            result.extend(list(self.names))

        d = toolz.merge(schema_methods(self.schema),
                        dshape_methods(self.dshape))
        result.extend(list(d))

        return sorted(set(result))

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            if key[0] == '_':
                raise
            if self.names and key in self.names:
                return self[key]
            d = toolz.merge(schema_methods(self.schema),
                            dshape_methods(self.dshape))
            if key in d:
                func = d[key]
                if func in method_properties:
                    return func(self)
                else:
                    return partial(func, self)
            else:
                raise AttributeError(key)


class Symbol(Expr):
    """
    Symbolic data

    >>> points = Symbol('points', '5 * 3 * {x: int, y: int}')
    """
    __slots__ = '_name', 'dshape'

    def __init__(self, name, dshape):
        self._name = name
        if isinstance(dshape, str):
            dshape = datashape.dshape(dshape)
        self.dshape = dshape


class ElemWise(Expr):
    """
    Elementwise operation
    """
    @property
    def dshape(self):
        return datashape.DataShape(*(self.child.dshape.shape
                                  + tuple(self.schema)))

class Field(ElemWise):
    """ A single field from an expression

    SELECT a
    FROM table

    >>> points = Symbol('points', '5 * 3 * {x: int32, y: int32}')
    >>> points.x.dshape
    dshape("5 * 3 * int32")
    """
    __slots__ = 'child', '_name'

    def __str__(self):
        return "%s['%s']" % (self.child, self.names[0])

    @property
    def expr(self):
        return ScalarSymbol(self._name, dtype=self.dtype)

    @property
    def dshape(self):
        shape = self.child.dshape.shape
        schema = self.child.schema[0].dict[self._name]

        shape = shape + schema.shape
        schema = (schema.measure,)
        return DataShape(*(shape + schema))


class Projection(ElemWise):
    """ Select fields from data

    SELECT a, b, c
    FROM table

    Examples
    --------

    >>> from blaze import TableSymbol
    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')
    >>> accounts[['name', 'amount']].schema
    dshape("{ name : string, amount : int32 }")

    See Also
    --------

    blaze.expr.core.Field
    """
    __slots__ = 'child', '_names'

    @property
    def names(self):
        return list(self._names)

    @property
    def schema(self):
        d = self.child.schema[0].dict
        return DataShape(Record([(name, d[name]) for name in self.names]))

    def __str__(self):
        return '%s[[%s]]' % (self.child,
                             ', '.join(["'%s'" % name for name in self.names]))

    def project(self, key):
        if isinstance(key, _strtypes) and key in self.names:
            return self.child[key]
        if isinstance(key, list) and set(key).issubset(set(self.names)):
            return self.child[key]
        raise ValueError("Column Mismatch: %s" % key)

    def get_field(self, fieldname):
        if fieldname in self.names:
            return Field(self.child, fieldname)
        raise ValueError("Field %s not found in columns %s" % (fieldname,
            self.names))


def projection(expr, names):
    return Projection(expr, tuple(names))

projection.__doc__ = Projection.__doc__


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
    >>> subs(t, {'balance': 'amount'}).names
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


schema_method_list = [
    ]

dshape_method_list = [
    ]

method_properties = set()

schema_methods = memoize(partial(select_functions, schema_method_list))
dshape_methods = memoize(partial(select_functions, dshape_method_list))
