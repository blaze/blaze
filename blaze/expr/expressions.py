from __future__ import absolute_import, division, print_function

import numbers
import toolz
import inspect
import datashape
import functools
from toolz import  concat, memoize, partial, first

from datashape import dshape, DataShape, Record, Var, Option, Unit
from datashape.predicates import isscalar, iscollection, isboolean, isrecord

from ..compatibility import _strtypes, builtins
from .core import *
from .method_dispatch import select_functions
from ..dispatch import dispatch

__all__ = ['Expr', 'ElemWise', 'Field', 'Symbol', 'discover', 'Projection',
           'projection', 'Selection', 'selection', 'Label', 'label', 'Map',
           'ReLabel', 'relabel', 'Apply']


class Expr(Node):
    """
    Symbolic expression of a computation

    All Blaze expressions (Join, By, Sort, ...) descend from this class.  It
    contains shared logic and syntax.  It in turn inherits from ``Node`` which
    holds all tree traversal logic
    """
    def _get_field(self, fieldname):
        if not isinstance(self.dshape.measure, Record):
            if fieldname == self._name:
                return self
            raise ValueError("Can not get field '%s' of non-record expression %s"
                    % (fieldname, self))
        return Field(self, fieldname)

    def __getitem__(self, key):
        if isinstance(key, _strtypes) and key in self.fields:
            return self._get_field(key)
        if isinstance(key, Expr) and iscollection(key.dshape):
            return selection(self, key)
        if (isinstance(key, list)
                and builtins.all(isinstance(k, _strtypes) for k in key)):
            if set(key).issubset(self.fields):
                return self._project(key)
            else:
                raise ValueError('Names %s not consistent with known names %s'
                        % (key, self.fields))
        raise ValueError("Not understood %s[%s]" % (self, key))

    def map(self, func, schema=None, name=None):
        return Map(self, func, schema, name)

    def _project(self, key):
        return projection(self, key)

    @property
    def schema(self):
        return datashape.dshape(self.dshape.measure)

    @property
    def fields(self):
        if isinstance(self.dshape.measure, Record):
            return self.dshape.measure.names
        if hasattr(self, '_name'):
            return [self._name]

    def _len(self):
        try:
            return int(self.dshape[0])
        except TypeError:
            raise ValueError('Can not determine length of table with the '
                    'following datashape: %s' % self.dshape)

    def __len__(self): # pragma: no cover
        return self._len()

    def __dir__(self):
        result = dir(type(self))
        if self.fields:
            result.extend(list(self.fields))

        d = toolz.merge(schema_methods(self.dshape.measure),
                        dshape_methods(self.dshape))
        result.extend(list(d))

        return sorted(set(result))

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            if self.fields and key in self.fields:
                if isscalar(self.dshape.measure): # t.foo.foo is t.foo
                    return self
                else:
                    return self[key]
            d = toolz.merge(schema_methods(self.dshape.measure),
                            dshape_methods(self.dshape))
            if key in d:
                func = d[key]
                if func in method_properties:
                    return func(self)
                else:
                    return functools.update_wrapper(partial(func, self), func)
            else:
                raise

class Symbol(Expr):
    """
    Symbolic data.  The leaf of a Blaze expression

    Example
    -------

    >>> points = Symbol('points', '5 * 3 * {x: int, y: int}')
    """
    __slots__ = '_name', 'dshape'
    __inputs__ = ()

    def __init__(self, name, dshape):
        self._name = name
        if isinstance(dshape, _strtypes):
            dshape = datashape.dshape(dshape)
        self.dshape = dshape

    def __str__(self):
        return self._name

    def resources(self):
        return dict()


class ElemWise(Expr):
    """
    Elementwise operation.

    The shape of this expression matches the shape of the child.
    """
    @property
    def dshape(self):
        return datashape.DataShape(*(self._child.dshape.shape
                                  + tuple(self.schema)))


class Field(ElemWise):
    """
    A single field from an expression

    Get a single field from an expression with record-type schema.  Collapses
    that record.  We store the name of the field in the ``_name`` attribute.

    SELECT a
    FROM table

    >>> points = Symbol('points', '5 * 3 * {x: int32, y: int32}')
    >>> points.x.dshape
    dshape("5 * 3 * int32")
    """
    __slots__ = '_child', '_name'

    def __str__(self):
        return "%s['%s']" % (self._child, self._name)

    @property
    def expr(self):
        return Symbol(self._name, datashape.DataShape(self.dshape.measure))

    @property
    def dshape(self):
        shape = self._child.dshape.shape
        schema = self._child.dshape.measure.dict[self._name]

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

    blaze.expr.expressions.Field
    """
    __slots__ = '_child', '_fields'

    @property
    def fields(self):
        return list(self._fields)

    @property
    def schema(self):
        d = self._child.schema[0].dict
        return DataShape(Record([(name, d[name]) for name in self.fields]))

    def __str__(self):
        return '%s[[%s]]' % (self._child,
                             ', '.join(["'%s'" % name for name in self.fields]))

    def _project(self, key):
        if isinstance(key, list) and set(key).issubset(set(self.fields)):
            return self._child[key]
        raise ValueError("Column Mismatch: %s" % key)

    def _get_field(self, fieldname):
        if fieldname in self.fields:
            return Field(self._child, fieldname)
        raise ValueError("Field %s not found in columns %s" % (fieldname,
            self.fields))


def projection(expr, names):
    if not isinstance(names, (tuple, list)):
        raise TypeError("Wanted list of strings, got %s" % names)
    if not set(names).issubset(expr.fields):
        raise ValueError("Mismatched names.  Asking for names %s "
                "where expression has names %s" % (names, expr.fields))
    return Projection(expr, tuple(names))
projection.__doc__ = Projection.__doc__


class Selection(Expr):
    """ Filter elements of expression based on predicate

    Examples
    --------

    >>> from blaze import TableSymbol
    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')
    >>> deadbeats = accounts[accounts['amount'] < 0]
    """
    __slots__ = '_child', 'predicate'

    def __str__(self):
        return "%s[%s]" % (self._child, self.predicate)

    @property
    def dshape(self):
        shape = list(self._child.dshape.shape)
        shape[0] = Var()
        return DataShape(*(shape + [self._child.dshape.measure]))


def selection(table, predicate):
    subexpr = common_subexpression(table, predicate)

    if not builtins.all(isinstance(node, (ElemWise, Symbol))
                        or node.isidentical(subexpr)
           for node in concat([path(predicate, subexpr),
                               path(table, subexpr)])):

        raise ValueError("Selection not properly matched with table:\n"
                   "child: %s\n"
                   "apply: %s\n"
                   "predicate: %s" % (subexpr, table, predicate))

    if not isboolean(predicate.dshape):
        raise TypeError("Must select over a boolean predicate.  Got:\n"
                        "%s[%s]" % (table, predicate))

    return table._subs({subexpr: Selection(subexpr, predicate)})

selection.__doc__ = Selection.__doc__


class Label(ElemWise):
    """ A Labeled expresion

    Examples
    --------

    >>> from blaze import TableSymbol
    >>> accounts = TableSymbol('accounts', '{name: string, amount: int}')

    >>> (accounts['amount'] * 100)._name
    'amount'

    >>> (accounts['amount'] * 100).label('new_amount')._name
    'new_amount'

    See Also
    --------

    blaze.expr.expressions.ReLabel
    """
    __slots__ = '_child', 'label'

    @property
    def schema(self):
        return self._child.schema

    @property
    def _name(self):
        return self.label

    def _get_field(self, key):
        if key[0] == self.fields[0]:
            return self
        else:
            raise ValueError("Column Mismatch: %s" % key)

def label(expr, lab):
    return Label(expr, lab)
label.__doc__ = Label.__doc__


class ReLabel(ElemWise):
    """
    Table with same content but with new labels

    Examples
    --------

    >>> from blaze import TableSymbol
    >>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
    >>> accounts.schema
    dshape("{ name : string, amount : int32 }")
    >>> accounts.relabel({'amount': 'balance'}).schema
    dshape("{ name : string, balance : int32 }")

    See Also
    --------

    blaze.expr.expressions.Label
    """
    __slots__ = '_child', 'labels'

    @property
    def schema(self):
        subs = dict(self.labels)
        d = self._child.dshape.measure.dict

        return DataShape(Record([[subs.get(name, name), dtype]
            for name, dtype in self._child.dshape.measure.parameters[0]]))


def relabel(child, labels):
    if isinstance(labels, dict):  # Turn dict into tuples
        labels = tuple(sorted(labels.items()))
    if isscalar(child.dshape.measure):
        if child._name == labels[0][0]:
            return child.label(labels[0][1])
        else:
            return child
    return ReLabel(child, labels)

relabel.__doc__ = ReLabel.__doc__


class Map(ElemWise):
    """ Map an arbitrary Python function across elements in a collection

    Examples
    --------

    >>> from datetime import datetime
    >>> from blaze import TableSymbol

    >>> t = TableSymbol('t', '{price: real, time: int64}')  # times as integers
    >>> datetimes = t['time'].map(datetime.utcfromtimestamp)

    Optionally provide extra schema information

    >>> datetimes = t['time'].map(datetime.utcfromtimestamp,
    ...                           schema='{time: datetime}')

    See Also
    --------

    blaze.expr.expresions.Apply
    """
    __slots__ = '_child', 'func', '_schema', '_name0'

    @property
    def schema(self):
        if self._schema:
            return dshape(self._schema)
        else:
            raise NotImplementedError("Schema of mapped column not known.\n"
                    "Please specify datashape keyword in .map method.\n"
                    "Example: t['columnname'].map(function, 'int64')")

    def label(self, name):
        assert isscalar(self.dshape.measure)
        return Map(self._child,
                   self.func,
                   self.schema,
                   name)

    @property
    def shape(self):
        return self._child.shape

    @property
    def ndim(self):
        return self._child.ndim

    @property
    def _name(self):
        if self._name0:
            return self._name0
        else:
            return self._child._name


class Apply(Expr):
    """ Apply an arbitrary Python function onto an expression

    Examples
    --------

    >>> from blaze import TableSymbol
    >>> t = TableSymbol('t', '{name: string, amount: int}')
    >>> h = Apply(hash, t)  # Hash value of resultant table

    Optionally provide extra datashape information

    >>> h = Apply(hash, t, dshape='real')

    Apply brings a function within the expression tree.
    The following transformation is often valid

    Before ``compute(Apply(f, expr), ...)``
    After  ``f(compute(expr, ...)``

    See Also
    --------

    blaze.expr.expressions.Map
    """
    __slots__ = '_child', 'func', '_dshape'

    def __init__(self, func, child, dshape=None):
        self._child = child
        self.func = func
        self._dshape = dshape

    @property
    def schema(self):
        if iscollection(self.dshape):
            return self.dshape.subshape[0]
        else:
            raise TypeError("Non-tabular datashape, %s" % self.dshape)

    @property
    def dshape(self):
        if self._dshape:
            return dshape(self._dshape)
        else:
            raise NotImplementedError("Datashape of arbitrary Apply not defined")


dshape_method_list = list()
schema_method_list = list()
method_properties = set()

dshape_methods = memoize(partial(select_functions, dshape_method_list))
schema_methods = memoize(partial(select_functions, schema_method_list))


def shape(expr):
    """ Shape of expression

    >>> Symbol('s', '3 * 5 * int32').shape
    (3, 5)
    """
    s = list(expr.dshape.shape)
    for i, elem in enumerate(s):
        try:
            s[i] = int(elem)
        except TypeError:
            pass

    return tuple(s)


def ndim(expr):
    """ Number of dimensions of expression

    >>> Symbol('s', '3 * var * int32').ndim
    2
    """
    return len(expr.shape)


dshape_method_list.extend([
    (iscollection, set([shape, ndim])),
    ])

schema_method_list.extend([
    (isscalar,  set([label, relabel])),
    (isrecord,  set([relabel])),
    ])

method_properties.update([shape, ndim])


@dispatch(Expr)
def discover(expr):
    return expr.dshape

