from __future__ import absolute_import, division, print_function

from collections import Mapping
from keyword import iskeyword
import re

import datashape
from datashape import (
    dshape,
    DataShape,
    Record,
    Var,
    Mono,
    Fixed,
    promote,
    Option,
    Null,
)
from datashape.predicates import isscalar, iscollection, isboolean, isrecord
import numpy as np
from odo.utils import copydoc
import toolz
from toolz import concat, memoize, partial, first
from toolz.curried import map, filter

from ..compatibility import _strtypes, builtins, boundmethod, PY2
from .core import Node, subs, common_subexpression, path
from .method_dispatch import select_functions
from ..dispatch import dispatch
from .utils import hashable_index, replace_slices, maxshape
from ..utils import attribute


__all__ = [
    'Apply',
    'Cast',
    'Coalesce',
    'Coerce',
    'ElemWise',
    'Expr',
    'Field',
    'Label',
    'Map',
    'Projection',
    'ReLabel',
    'Selection',
    'SimpleSelection',
    'Slice',
    'Symbol',
    'apply',
    'cast',
    'coalesce',
    'coerce',
    'discover',
    'label',
    'ndim',
    'projection',
    'relabel',
    'selection',
    'shape',
    'symbol',
]


def isvalid_identifier(s):
    """Check whether a string is a valid Python identifier

    Examples
    --------
    >>> isvalid_identifier('Hello')
    True
    >>> isvalid_identifier('Hello world')
    False
    >>> isvalid_identifier('Helloworld!')
    False
    >>> isvalid_identifier('1a')
    False
    >>> isvalid_identifier('a1')
    True
    >>> isvalid_identifier('for')
    False
    >>> isvalid_identifier(None)
    False
    """
    # the re module compiles and caches regexs so no need to compile it
    return (s is not None and not iskeyword(s) and
            re.match(r'^[_a-zA-Z][_a-zA-Z0-9]*$', s) is not None)


def valid_identifier(s):
    """Rewrite a string to be a valid identifier if it contains
    >>> valid_identifier('hello')
    'hello'
    >>> valid_identifier('hello world')
    'hello_world'
    >>> valid_identifier('hello.world')
    'hello_world'
    >>> valid_identifier('hello-world')
    'hello_world'
    >>> valid_identifier(None)
    >>> valid_identifier('1a')
    """
    if isinstance(s, _strtypes):
        if not s or s[0].isdigit():
            return
        return s.replace(' ', '_').replace('.', '_').replace('-', '_')
    return s


class Expr(Node):
    """
    Symbolic expression of a computation

    All Blaze expressions (Join, By, Sort, ...) descend from this class.  It
    contains shared logic and syntax.  It in turn inherits from ``Node`` which
    holds all tree traversal logic
    """
    __slots__ = '_hash', '__weakref__', '__dict__'

    def _get_field(self, fieldname):
        if not isinstance(self.dshape.measure, (Record, datashape.Map)):
            if fieldname == self._name:
                return self
            raise ValueError(
                "Can not get field '%s' of non-record expression %s" %
                (fieldname, self))
        return Field(self, fieldname)

    def __getitem__(self, key):
        if isinstance(key, _strtypes) and key in self.fields:
            return self._get_field(key)
        elif isinstance(key, Expr) and iscollection(key.dshape):
            return selection(self, key)
        elif (isinstance(key, list) and
              builtins.all(isinstance(k, _strtypes) for k in key)):
            if set(key).issubset(self.fields):
                return self._project(key)
            else:
                raise ValueError('Names %s not consistent with known names %s'
                                 % (key, self.fields))
        elif (isinstance(key, tuple) and
              all(isinstance(k, (int, slice, type(None), list, np.ndarray))
              for k in key)):
            return sliceit(self, key)
        elif isinstance(key, (slice, int, type(None), list, np.ndarray)):
            return sliceit(self, (key,))
        raise ValueError("Not understood %s[%s]" % (self, key))

    def map(self, func, schema=None, name=None):
        return Map(self, func, schema, name)

    def _project(self, key):
        return projection(self, key)

    @attribute
    def schema(self):
        try:
            m = self._schema
        except AttributeError:
            pass
        else:
            return m()

        self.schema = schema = datashape.dshape(self.dshape.measure)
        return schema

    @attribute
    def dshape(self):
        self.dshape = dshape = self._dshape()
        return dshape

    @property
    def fields(self):
        measure = self.dshape.measure
        if isinstance(self.dshape.measure, Option):
            measure = measure.ty
        if isinstance(measure, Record):
            return measure.names
        elif isinstance(measure, datashape.Map):
            if not isrecord(self.dshape.measure.value):
                raise TypeError('Foreign key must reference a '
                                'Record datashape')
            return measure.value.names
        name = getattr(self, '_name', None)
        if name is not None:
            return [self._name]
        return []

    def _len(self):
        try:
            return int(self.dshape[0])
        except TypeError:
            raise ValueError('Can not determine length of table with the '
                             'following datashape: %s' % self.dshape)

    def __len__(self):  # pragma: no cover
        return self._len()

    def __iter__(self):
        raise NotImplementedError(
            'Iteration over expressions is not supported.\n'
            'Iterate over computed result instead, e.g. \n'
            "\titer(expr)           # don't do this\n"
            "\titer(compute(expr))  # do this instead")

    def __dir__(self):
        result = dir(type(self))
        if (isrecord(self.dshape.measure) or
            isinstance(self.dshape.measure, datashape.Map) and
                self.fields):
            result.extend(map(valid_identifier, self.fields))

        result.extend(toolz.merge(schema_methods(self.dshape.measure),
                                  dshape_methods(self.dshape)))

        return sorted(set(filter(isvalid_identifier, result)))

    def __getattr__(self, key):
        assert key != '_hash', \
            '%s expressions should set _hash in __init__' % type(self).__name__
        try:
            result = object.__getattribute__(self, key)
        except AttributeError:
            fields = dict(zip(map(valid_identifier, self.fields), self.fields))

            # prefer the method if there's a field with the same name
            methods = toolz.merge(
                schema_methods(self.dshape.measure),
                dshape_methods(self.dshape)
            )
            if key in methods:
                func = methods[key]
                if func in method_properties:
                    result = func(self)
                else:
                    result = boundmethod(func, self)
            elif self.fields and key in fields:
                if isscalar(self.dshape.measure):  # t.foo.foo is t.foo
                    result = self
                else:
                    result = self[fields[key]]
            else:
                raise

        # cache the attribute lookup, getattr will not be invoked again.
        setattr(self, key, result)
        return result

    @property
    def _name(self):
        measure = self.dshape.measure
        if len(self._inputs) == 1 and isscalar(getattr(measure, 'key',
                                                       measure)):
            child_measure = self._child.dshape.measure
            if isscalar(getattr(child_measure, 'key', child_measure)):
                return self._child._name

    def __enter__(self):
        """ Enter context """
        return self

    def __exit__(self, *args):
        """ Exit context

        Close any open resource if we are called in context
        """
        for value in self._resources().values():
            try:
                value.close()
            except AttributeError:
                pass
        return True

_symbol_cache = dict()


def _symbol_key(args, kwargs):
    if len(args) == 1:
        name, = args
        ds = None
        token = 0
    if len(args) == 2:
        name, ds = args
        token = 0
    elif len(args) == 3:
        name, ds, token = args
        token = token or 0
    ds = kwargs.get('dshape', ds)
    token = kwargs.get('token', token)
    ds = dshape(ds)
    return (name, ds, token)


def sanitized_dshape(dshape, width=50):
    pretty_dshape = datashape.pprint(dshape, width=width).replace('\n', '')
    if len(pretty_dshape) > width:
        pretty_dshape = "{}...".format(pretty_dshape[:width])
    return pretty_dshape


class Symbol(Expr):
    """
    Symbolic data.  The leaf of a Blaze expression

    Examples
    --------
    >>> points = symbol('points', '5 * 3 * {x: int, y: int}')
    >>> points
    <`points` symbol; dshape='5 * 3 * {x: int32, y: int32}'>
    >>> points.dshape
    dshape("5 * 3 * {x: int32, y: int32}")
    """
    __slots__ = '_hash', '_name', 'dshape', '_token'
    __inputs__ = ()

    def __init__(self, name, dshape, token=0):
        self._name = name
        if isinstance(dshape, _strtypes):
            dshape = datashape.dshape(dshape)
        if isinstance(dshape, Mono) and not isinstance(dshape, DataShape):
            dshape = DataShape(dshape)
        self.dshape = dshape
        self._token = token
        self._hash = None

    def __repr__(self):
        fmt = "<`{}` symbol; dshape='{}'>"
        return fmt.format(self._name, sanitized_dshape(self.dshape))

    def __str__(self):
        return self._name or ''

    def _resources(self):
        return dict()


@memoize(cache=_symbol_cache, key=_symbol_key)
@copydoc(Symbol)
def symbol(name, dshape, token=None):
    return Symbol(name, dshape, token=token or 0)


@dispatch(Symbol, Mapping)
def _subs(o, d):
    """ Subs symbols using symbol function

    Supports caching"""
    newargs = [subs(arg, d) for arg in o._args]
    return symbol(*newargs)


class ElemWise(Expr):
    """
    Elementwise operation.

    The shape of this expression matches the shape of the child.
    """
    def _dshape(self):
        return datashape.DataShape(
            *(self._child.dshape.shape + tuple(self.schema))
        )


class Field(ElemWise):
    """
    A single field from an expression.

    Get a single field from an expression with record-type schema.
    We store the name of the field in the ``_name`` attribute.

    Examples
    --------
    >>> points = symbol('points', '5 * 3 * {x: int32, y: int32}')
    >>> points.x.dshape
    dshape("5 * 3 * int32")

    For fields that aren't valid Python identifiers, use ``[]`` syntax:

    >>> points = symbol('points', '5 * 3 * {"space station": float64}')
    >>> points['space station'].dshape
    dshape("5 * 3 * float64")
    """
    __slots__ = '_hash', '_child', '_name'

    def __str__(self):
        fmt = '%s.%s' if isvalid_identifier(self._name) else '%s[%r]'
        return fmt % (self._child, self._name)

    @property
    def _expr(self):
        return symbol(self._name, datashape.DataShape(self.dshape.measure))

    def _dshape(self):
        shape = self._child.dshape.shape
        measure = self._child.dshape.measure

        # TODO: is this too special-case-y?
        schema = getattr(measure, 'value', measure).dict[self._name]

        shape = shape + schema.shape
        schema = (schema.measure,)
        return DataShape(*(shape + schema))


class Projection(ElemWise):
    """Select a subset of fields from data.

    Examples
    --------
    >>> accounts = symbol('accounts',
    ...                   'var * {name: string, amount: int, id: int}')
    >>> accounts[['name', 'amount']].schema
    dshape("{name: string, amount: int32}")
    >>> accounts[['name', 'amount']]
    accounts[['name', 'amount']]

    See Also
    --------
    blaze.expr.expressions.Field
    """
    __slots__ = '_hash', '_child', '_fields'

    @property
    def fields(self):
        return list(self._fields)

    def _schema(self):
        measure = self._child.schema.measure
        d = getattr(measure, 'value', measure).dict
        return DataShape(Record((name, d[name]) for name in self.fields))

    def __str__(self):
        return '%s[%s]' % (self._child, self.fields)

    def _project(self, key):
        if isinstance(key, list) and set(key).issubset(set(self.fields)):
            return self._child[key]
        raise ValueError("Column Mismatch: %s" % key)

    def _get_field(self, fieldname):
        if fieldname in self.fields:
            return Field(self._child, fieldname)
        raise ValueError("Field %s not found in columns %s" % (fieldname,
                                                               self.fields))


@copydoc(Projection)
def projection(expr, names):
    if not names:
        raise ValueError("Projection with no names")
    if not isinstance(names, (tuple, list)):
        raise TypeError("Wanted list of strings, got %s" % names)
    if not set(names).issubset(expr.fields):
        raise ValueError("Mismatched names. Asking for names %s "
                         "where expression has names %s" %
                         (names, expr.fields))
    return Projection(expr, tuple(names))


def sanitize_index_lists(ind):
    """ Handle lists/arrays of integers/bools as indexes

    >>> sanitize_index_lists([2, 3, 5])
    [2, 3, 5]
    >>> sanitize_index_lists([True, False, True, False])
    [0, 2]
    >>> sanitize_index_lists(np.array([1, 2, 3]))
    [1, 2, 3]
    >>> sanitize_index_lists(np.array([False, True, True]))
    [1, 2]
    """
    if not isinstance(ind, (list, np.ndarray)):
        return ind
    if isinstance(ind, np.ndarray):
        ind = ind.tolist()
    if isinstance(ind, list) and ind and isinstance(ind[0], bool):
        ind = [a for a, b in enumerate(ind) if b]
    return ind


def sliceit(child, index):
    index2 = tuple(map(sanitize_index_lists, index))
    index3 = hashable_index(index2)
    s = Slice(child, index3)
    hash(s)
    return s


class Slice(Expr):
    """Elements `start` until `stop`. On many backends, a `step` parameter
    is also allowed.

    Examples
    --------
    >>> from blaze import symbol
    >>> accounts = symbol('accounts', 'var * {name: string, amount: int}')
    >>> accounts[2:7].dshape
    dshape("5 * {name: string, amount: int32}")
    >>> accounts[2:7:2].dshape
    dshape("3 * {name: string, amount: int32}")
    """
    __slots__ = '_hash', '_child', '_index'

    def _dshape(self):
        return self._child.dshape.subshape[self.index]

    @property
    def index(self):
        return replace_slices(self._index)

    def __str__(self):
        if isinstance(self.index, tuple):
            index = ', '.join(map(str, self._index))
        else:
            index = str(self._index)
        return '%s[%s]' % (self._child, index)


class Selection(Expr):
    """ Filter elements of expression based on predicate

    Examples
    --------

    >>> accounts = symbol('accounts',
    ...                   'var * {name: string, amount: int, id: int}')
    >>> deadbeats = accounts[accounts.amount < 0]
    """
    __slots__ = '_hash', '_child', 'predicate'
    __inputs__ = '_child', 'predicate'

    @property
    def _name(self):
        return self._child._name

    def __str__(self):
        return "%s[%s]" % (self._child, self.predicate)

    def _dshape(self):
        shape = list(self._child.dshape.shape)
        shape[0] = Var()
        return DataShape(*(shape + [self._child.dshape.measure]))


class SimpleSelection(Selection):
    """Internal selection class that does not treat the predicate as an input.
    """
    __slots__ = Selection.__slots__
    __inputs__ = '_child',


@copydoc(Selection)
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


class Label(ElemWise):
    """An expression with a name.

    Examples
    --------
    >>> accounts = symbol('accounts', 'var * {name: string, amount: int}')
    >>> expr = accounts.amount * 100
    >>> expr._name
    'amount'
    >>> expr.label('new_amount')._name
    'new_amount'

    See Also
    --------
    blaze.expr.expressions.ReLabel
    """
    __slots__ = '_hash', '_child', 'label'

    def _schema(self):
        return self._child.schema

    @property
    def _name(self):
        return self.label

    def _get_field(self, key):
        if key[0] == self.fields[0]:
            return self
        raise ValueError("Column Mismatch: %s" % key)

    def __str__(self):
        return 'label(%s, %r)' % (self._child, self.label)


@copydoc(Label)
def label(expr, lab):
    if expr._name == lab:
        return expr
    return Label(expr, lab)


class ReLabel(ElemWise):
    """
    Table with same content but with new labels

    Examples
    --------
    >>> accounts = symbol('accounts', 'var * {name: string, amount: int}')
    >>> accounts.schema
    dshape("{name: string, amount: int32}")
    >>> accounts.relabel(amount='balance').schema
    dshape("{name: string, balance: int32}")
    >>> accounts.relabel(not_a_column='definitely_not_a_column')
    Traceback (most recent call last):
        ...
    ValueError: Cannot relabel non-existent child fields: {'not_a_column'}
    >>> s = symbol('s', 'var * {"0": int64}')
    >>> s.relabel({'0': 'foo'})
    s.relabel({'0': 'foo'})
    >>> s.relabel(0='foo') # doctest: +SKIP
    Traceback (most recent call last):
        ...
    SyntaxError: keyword can't be an expression

    Notes
    -----
    When names are not valid Python names, such as integers or string with
    spaces, you must pass a dictionary to ``relabel``. For example

    .. code-block:: python

       >>> s = symbol('s', 'var * {"0": int64}')
       >>> s.relabel({'0': 'foo'})
       s.relabel({'0': 'foo'})
       >>> t = symbol('t', 'var * {"whoo hoo": ?float32}')
       >>> t.relabel({"whoo hoo": 'foo'})
       t.relabel({'whoo hoo': 'foo'})

    See Also
    --------
    blaze.expr.expressions.Label
    """
    __slots__ = '_hash', '_child', 'labels'

    def _schema(self):
        subs = dict(self.labels)
        param = self._child.dshape.measure.parameters[0]
        return DataShape(Record([[subs.get(name, name), dtype]
                                 for name, dtype in param]))

    def __str__(self):
        labels = self.labels
        if all(map(isvalid_identifier, map(first, labels))):
            rest = ', '.join('%s=%r' % l for l in labels)
        else:
            rest = '{%s}' % ', '.join('%r: %r' % l for l in labels)
        return '%s.relabel(%s)' % (self._child, rest)


@copydoc(ReLabel)
def relabel(child, labels=None, **kwargs):
    labels = {k: v
              for k, v in toolz.merge(labels or {}, kwargs).items() if k != v}
    label_keys = set(labels)
    fields = child.fields
    if not label_keys.issubset(fields):
        non_existent_fields = label_keys.difference(fields)
        raise ValueError("Cannot relabel non-existent child fields: {%s}" %
                         ', '.join(map(repr, non_existent_fields)))
    if not labels:
        return child
    if isinstance(labels, Mapping):  # Turn dict into tuples
        labels = tuple(sorted(labels.items()))
    if isscalar(child.dshape.measure):
        if child._name == labels[0][0]:
            return child.label(labels[0][1])
        else:
            return child
    return ReLabel(child, labels)


class Map(ElemWise):
    """ Map an arbitrary Python function across elements in a collection

    Examples
    --------
    >>> from datetime import datetime

    >>> t = symbol('t', 'var * {price: real, time: int64}')  # times as integers
    >>> datetimes = t.time.map(datetime.utcfromtimestamp)

    Optionally provide extra schema information

    >>> datetimes = t.time.map(datetime.utcfromtimestamp,
    ...                           schema='{time: datetime}')

    See Also
    --------
    blaze.expr.expresions.Apply
    """
    __slots__ = '_hash', '_child', 'func', '_asschema', '_name0'

    def _schema(self):
        if self._asschema:
            return dshape(self._asschema)
        else:
            raise NotImplementedError("Schema of mapped column not known.\n"
                                      "Please specify datashape keyword in "
                                      ".map method.\nExample: "
                                      "t.columnname.map(function, 'int64')")

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


if PY2:
    copydoc(Map, Expr.map.im_func)
else:
    copydoc(Map, Expr.map)


class Apply(Expr):
    """ Apply an arbitrary Python function onto an expression

    Examples
    --------

    >>> t = symbol('t', 'var * {name: string, amount: int}')
    >>> h = t.apply(hash, dshape='int64')  # Hash value of resultant dataset

    You must provide the datashape of the result with the ``dshape=`` keyword.
    For datashape examples see
    http://datashape.pydata.org/grammar.html#some-simple-examples

    If using a chunking backend and your operation may be safely split and
    concatenated then add the ``splittable=True`` keyword argument

    >>> t.apply(f, dshape='...', splittable=True) # doctest: +SKIP

    See Also
    --------

    blaze.expr.expressions.Map
    """
    __slots__ = '_hash', '_child', 'func', '_asdshape', '_splittable'

    def _schema(self):
        if iscollection(self.dshape):
            return self.dshape.subshape[0]
        else:
            raise TypeError("Non-tabular datashape, %s" % self.dshape)

    def _dshape(self):
        return dshape(self._asdshape)


@copydoc(Apply)
def apply(expr, func, dshape, splittable=False):
    return Apply(expr, func, datashape.dshape(dshape), splittable)


class Coerce(ElemWise):
    """Coerce an expression to a different type.

    Examples
    --------
    >>> t = symbol('t', '100 * float64')
    >>> t.coerce(to='int64')
    t.coerce(to='int64')
    >>> t.coerce('float32')
    t.coerce(to='float32')
    >>> t.coerce('int8').dshape
    dshape("100 * int8")
    """
    __slots__ = '_hash', '_child', 'to'

    def _schema(self):
        return self.to

    def __str__(self):
        return '%s.coerce(to=%r)' % (self._child, str(self.schema))


@copydoc(Coerce)
def coerce(expr, to):
    return Coerce(expr, dshape(to) if isinstance(to, _strtypes) else to)


class Cast(Expr):
    """Cast an expression to a different type.

    This is only an expression time operation.

    Examples
    --------
    >>> s = symbol('s', '?int64')
    >>> s.cast('?int32').dshape
    dshape("?int32")

    # Cast to correct mislabeled optionals
    >>> s.cast('int64').dshape
    dshape("int64")

    # Cast to give concrete dimension length
    >>> t = symbol('t', 'var * float32')
    >>> t.cast('10 * float32').dshape
    dshape("10 * float32")
    """
    __slots__ = '_hash', '_child', 'to'

    def _dshape(self):
        return self.to

    def __str__(self):
        return 'cast(%s, to=%r)' % (self._child, str(self.to))


@copydoc(Cast)
def cast(expr, to):
    return Cast(expr, dshape(to) if isinstance(to, _strtypes) else to)


Expr.cast = cast  # method of all exprs


def binop_name(expr):
    if not isscalar(expr.dshape.measure):
        return None
    l = getattr(expr.lhs, '_name', None)
    r = getattr(expr.rhs, '_name', None)
    if bool(l) ^ bool(r):
        return l or r
    elif l == r:
        return l

    return None


def binop_inputs(expr):
    if isinstance(expr.lhs, Expr):
        yield expr.lhs
    if isinstance(expr.rhs, Expr):
        yield expr.rhs


class Coalesce(Expr):
    """SQL like coalesce.

    coalesce(a, b) = {
        a if a is not NULL
        b otherwise
    }

    Examples
    --------
    >>> coalesce(1, 2)
    1
    >>> coalesce(1, None)
    1
    >>> coalesce(None, 2)
    2
    >>> coalesce(None, None) is None
    True
    """
    __slots__ = '_hash', 'lhs', 'rhs', 'dshape'
    __inputs__ = 'lhs', 'rhs'

    def __str__(self):
        return 'coalesce(%s, %s)' % (self.lhs, self.rhs)

    _name = property(binop_name)

    @property
    def _inputs(self):
        return tuple(binop_inputs(self))


@copydoc(Coalesce)
def coalesce(a, b):
    a_dshape = discover(a)
    a_measure = a_dshape.measure
    isoption = isinstance(a_measure, Option)
    if isoption:
        a_measure = a_measure.ty
    isnull = isinstance(a_measure, Null)
    if isnull:
        # a is always null, this is just b
        return b

    if not isoption:
        # a is not an option, this is just a
        return a

    b_dshape = discover(b)
    return Coalesce(a, b, DataShape(*(
        maxshape((a_dshape.shape, b_dshape.shape)) +
        (promote(a_measure, b_dshape.measure),)
    )))


dshape_method_list = list()
schema_method_list = list()
method_properties = set()

dshape_methods = memoize(partial(select_functions, dshape_method_list))
schema_methods = memoize(partial(select_functions, schema_method_list))


@dispatch(DataShape)
def shape(ds):
    s = ds.shape
    s = tuple(int(d) if isinstance(d, Fixed) else d for d in s)
    return s


@dispatch(object)
def shape(expr):
    """ Shape of expression

    >>> symbol('s', '3 * 5 * int32').shape
    (3, 5)

    Works on anything discoverable

    >>> shape([[1, 2], [3, 4]])
    (2, 2)
    """
    s = list(discover(expr).shape)
    for i, elem in enumerate(s):
        try:
            s[i] = int(elem)
        except TypeError:
            pass

    return tuple(s)


def ndim(expr):
    """ Number of dimensions of expression

    >>> symbol('s', '3 * var * int32').ndim
    2
    """
    return len(shape(expr))


dshape_method_list.extend([
    (lambda ds: True, set([apply])),
    (iscollection, set([shape, ndim])),
    (lambda ds: iscollection(ds) and isscalar(ds.measure), set([coerce]))
])

schema_method_list.extend([
    (isscalar, set([label, relabel, coerce])),
    (isrecord, set([relabel])),
    (lambda ds: isinstance(ds, Option), {coalesce}),
])

method_properties.update([shape, ndim])


@dispatch(Expr)
def discover(expr):
    return expr.dshape
