from __future__ import absolute_import, division, print_function

import numbers
import toolz
import inspect
import datashape
from toolz import  concat, memoize, partial, first

from datashape import dshape, DataShape, Record, Var, Option, Unit
from datashape.predicates import isscalar, iscollection

from ..compatibility import _strtypes, builtins
from .core import *
from .method_dispatch import select_functions
from ..dispatch import dispatch

__all__ = ['Expr', 'ElemWise', 'Field', 'Symbol', 'discover', 'Projection',
           'projection', 'Selection', 'selection', 'Label', 'label', 'Map']


class Expr(Node):

    def get_field(self, fieldname):
        if not isinstance(self.dshape.measure, Record):
            if fieldname == self._name:
                return self
            raise ValueError("Can not get field '%s' of non-record expression %s"
                    % (fieldname, self))
        return Field(self, fieldname)

    def __getitem__(self, key):
        if isinstance(key, _strtypes) and key in self.fields:
            return self.get_field(key)
        if isinstance(key, Expr) and iscollection(key.dshape):
            from .expr import selection
            return selection(self, key)
        if (isinstance(key, list)
                and builtins.all(isinstance(k, _strtypes) for k in key)):
            if set(key).issubset(self.fields):
                return self.project(key)
            else:
                raise ValueError('Names %s not consistent with known names %s'
                        % (key, self.fields))
        raise ValueError("Not understood %s[%s]" % (self, key))

    def map(self, func, schema=None, name=None):
        from .expr import Map
        return Map(self, func, schema, name)

    def project(self, key):
        return projection(self, key)

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
                    return partial(func, self)
            else:
                raise

class Symbol(Expr):
    """
    Symbolic data

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
        return "%s['%s']" % (self.child, self._name)

    @property
    def expr(self):
        return Symbol(self._name, datashape.DataShape(self.dshape.measure))

    @property
    def dshape(self):
        shape = self.child.dshape.shape
        schema = self.child.dshape.measure.dict[self._name]

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

    blaze.expr.expr.Field
    """
    __slots__ = 'child', '_fields'

    @property
    def fields(self):
        return list(self._fields)

    @property
    def schema(self):
        d = self.child.schema[0].dict
        return DataShape(Record([(name, d[name]) for name in self.fields]))

    def __str__(self):
        return '%s[[%s]]' % (self.child,
                             ', '.join(["'%s'" % name for name in self.fields]))

    def project(self, key):
        if isinstance(key, list) and set(key).issubset(set(self.fields)):
            return self.child[key]
        raise ValueError("Column Mismatch: %s" % key)

    def get_field(self, fieldname):
        if fieldname in self.fields:
            return Field(self.child, fieldname)
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
    __slots__ = 'child', 'predicate'

    def __str__(self):
        return "%s[%s]" % (self.child, self.predicate)

    @property
    def dshape(self):
        shape = list(self.child.dshape.shape)
        shape[0] = Var()
        return DataShape(*(shape + [self.child.dshape.measure]))


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

    if predicate.dtype != dshape('bool'):
        raise TypeError("Must select over a boolean predicate.  Got:\n"
                        "%s[%s]" % (table, predicate))

    return table.subs({subexpr: Selection(subexpr, predicate)})

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

    blaze.expr.table.ReLabel
    """
    __slots__ = 'child', 'label'

    @property
    def schema(self):
        return self.child.schema

    @property
    def _name(self):
        return self.label

    def get_field(self, key):
        if key[0] == self.fields[0]:
            return self
        else:
            raise ValueError("Column Mismatch: %s" % key)

def label(expr, lab):
    return Label(expr, lab)
label.__doc__ = Label.__doc__


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

    blaze.expr.table.Apply
    """
    __slots__ = 'child', 'func', '_schema', '_name0'

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
        return Map(self.child,
                   self.func,
                   self.schema,
                   name)

    @property
    def shape(self):
        return self.child.shape

    @property
    def ndim(self):
        return self.child.ndim

    @property
    def _name(self):
        if self._name0:
            return self._name0
        else:
            return self.child._name


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

method_properties.update([shape, ndim])


@dispatch(Expr)
def discover(expr):
    return expr.dshape

