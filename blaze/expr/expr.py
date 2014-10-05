from __future__ import absolute_import, division, print_function

import numbers
import toolz
import inspect
import functools
import datashape
from datashape import dshape, DataShape, Record, Var, Option, Unit

from toolz import unique, concat, memoize, partial, first
import toolz
from blaze.compatibility import _strtypes, builtins

from . import scalar
from .core import Expr, common_subexpression, path
from .scalar import ScalarSymbol, Number
from .scalar import (Eq, Ne, Lt, Le, Gt, Ge, Add, Mult, Div, Sub, Pow, Mod, Or,
                     And, USub, Not, eval_str, FloorDiv, NumberInterface)
from datashape.predicates import isscalar, iscollection
from .method_dispatch import select_functions
from ..dispatch import dispatch

__all__ = ['Collection', 'Projection', 'projection', 'Selection', 'selection', 'broadcast',
'Broadcast', 'Label', 'label', 'ElemWise', 'Field', 'Symbol']


class Collection(Expr):
    __inputs__ = 'child',

    def _len(self):
        try:
            return int(self.dshape[0])
        except TypeError:
            raise ValueError('Can not determine length of table with the '
                    'following datashape: %s' % self.dshape)

    def __len__(self): # pragma: no cover
        return self._len()

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
        if isinstance(key, Collection):
            return selection(self, key)
        if (isinstance(key, list)
                and builtins.all(isinstance(k, _strtypes) for k in key)):
            if set(key).issubset(self.fields):
                return self.project(key)
            else:
                raise ValueError('Names %s not consistent with known names %s'
                        % (key, self.fields))
        raise ValueError("Not understood %s[%s]" % (self, key))

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

    def __dir__(self):
        result = dir(type(self))
        if self.fields:
            result.extend(list(self.fields))

        d = toolz.merge(schema_methods(self.schema),
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
            d = toolz.merge(schema_methods(self.schema),
                            dshape_methods(self.dshape))
            if key in d:
                func = d[key]
                if func in method_properties:
                    return func(self)
                else:
                    return partial(func, self)
            else:
                raise

    def __eq__(self, other):
        if self.isidentical(other) is True:
            return True
        if (isscalar(self.dshape.measure) and
                (not isinstance(other, Expr)
                     or isscalar(other.dshape.measure))):
            return broadcast(Eq, self, other)
        else:
            return self.isidentical(other)

    def map(self, func, schema=None, name=None):
        return Map(self, func, schema, name)


class Symbol(Collection):
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


class ElemWise(Collection):
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
        from .scalar import ScalarSymbol
        return ScalarSymbol(self._name, dtype=self.dtype)

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

    blaze.expr.core.Field
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


class Selection(Collection):
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


def _expr_child(col):
    """ Expr and child of field

    Examples
    --------

    >>> from blaze import TableSymbol
    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')
    >>> _expr_child(accounts['name'])
    (name, accounts)

    Helper function for ``broadcast``
    """
    if isinstance(col, (Broadcast, Field)):
        return col.expr, col.child
    elif isinstance(col, Label):
        return _expr_child(col.child)
    else:
        return col, None


def broadcast(op, *column_inputs):
    """ Broadcast scalar operation across multiple fields

    Parameters
    ----------
    op : Scalar Operation like Add, Mult, Sin, Exp

    column_inputs : either Column, Broadcast or constant (like 1, 1.0, '1')

    Examples
    --------

    >>> from blaze import TableSymbol
    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')

    >>> broadcast(Add, accounts['amount'], 100)
    accounts['amount'] + 100

    Fuses operations down into ScalarExpr level

    >>> broadcast(Mult, 2, (accounts['amount'] + 100))
    2 * (accounts['amount'] + 100)
    """
    expr_inputs = []
    children = set()

    for col in column_inputs:
        expr, child = _expr_child(col)
        expr_inputs.append(expr)
        if child:
            children.add(child)

    if not len(children) == 1:
        raise ValueError("All inputs must be from same Table.\n"
                         "Saw the following tables: %s"
                         % ', '.join(map(str, children)))

    if hasattr(op, 'op'):
        expr = op.op(*expr_inputs)
    else:
        expr = op(*expr_inputs)

    return Broadcast(first(children), expr)


class Broadcast(ElemWise):
    """ Apply Scalar Expression onto columns of data

    Parameters
    ----------

    child : TableExpr
    expr : ScalarExpr
        The names of the varibles within the scalar expr must match the columns
        of the child.  Use ``Column.scalar_variable`` to generate the
        appropriate ScalarSymbol

    Examples
    --------

    >>> from blaze import TableSymbol, Add
    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')

    >>> expr = Add(accounts['amount'].expr, 100)
    >>> Broadcast(accounts, expr)
    accounts['amount'] + 100

    See Also
    --------

    blaze.expr.table.broadcast
    """
    __slots__ = 'child', 'expr'

    @property
    def _name(self):
        names = [x._name for x in self.expr.traverse()
                         if isinstance(x, ScalarSymbol)]
        if len(names) == 1 and not isinstance(self.expr.dshape[0], Record):
            return names[0]

    @property
    def dshape(self):
        return DataShape(*(self.child.shape + (self.expr.dshape.measure,)))

    def __str__(self):
        columns = self.active_columns()
        newcol = lambda c: "%s['%s']" % (self.child, c)
        return eval_str(self.expr.subs(dict(zip(columns,
                                                map(newcol, columns)))))

    def active_columns(self):
        return sorted(unique(x._name for x in self.traverse()
                                    if isinstance(x, ScalarSymbol)))


sqrt = partial(broadcast, scalar.sqrt)

sin = partial(broadcast, scalar.sin)
cos = partial(broadcast, scalar.cos)
tan = partial(broadcast, scalar.tan)
sinh = partial(broadcast, scalar.sinh)
cosh = partial(broadcast, scalar.cosh)
tanh = partial(broadcast, scalar.tanh)
acos = partial(broadcast, scalar.acos)
acosh = partial(broadcast, scalar.acosh)
asin = partial(broadcast, scalar.asin)
asinh = partial(broadcast, scalar.asinh)
atan = partial(broadcast, scalar.atan)
atanh = partial(broadcast, scalar.atanh)

exp = partial(broadcast, scalar.exp)
log = partial(broadcast, scalar.log)
expm1 = partial(broadcast, scalar.expm1)
log10 = partial(broadcast, scalar.log10)
log1p = partial(broadcast, scalar.log1p)

radians = partial(broadcast, scalar.radians)
degrees = partial(broadcast, scalar.degrees)

ceil = partial(broadcast, scalar.ceil)
floor = partial(broadcast, scalar.floor)
trunc = partial(broadcast, scalar.trunc)

def isnan(expr):
    return broadcast(scalar.isnan, expr)

def _ne(a, b):
    return broadcast(Ne, a, b)

def _lt(a, b):
    return broadcast(Lt, a, b)

def _le(a, b):
    return broadcast(Le, a, b)

def _gt(a, b):
    return broadcast(Gt, a, b)

def _ge(a, b):
    return broadcast(Ge, a, b)

def _add(a, b):
    return broadcast(Add, a, b)

def _radd(a, b):
    return broadcast(Add, b, a)

def _mul(a, b):
    return broadcast(Mult, a, b)

def _rmul(a, b):
    return broadcast(Mult, b, a)

def _div(a, b):
    return broadcast(Div, a, b)

def _rdiv(a, b):
    return broadcast(Div, b, a)

def _floordiv(a, b):
    return broadcast(FloorDiv, a, b)

def _rfloordiv(a, b):
    return broadcast(FloorDiv, b, a)

def _sub(a, b):
    return broadcast(Sub, a, b)

def _rsub(a, b):
    return broadcast(Sub, b, a)

def _pow(a, b):
    return broadcast(Pow, a, b)

def _rpow(a, b):
    return broadcast(Pow, b, a)

def _mod(a, b):
    return broadcast(Mod, a, b)

def _rmod(a, b):
    return broadcast(Mod, b, a)

def _or(a, b):
    return broadcast(Or, a, b)

def _ror(a, b):
    return broadcast(Or, b, a)

def _and(a, b):
    return broadcast(And, a, b)

def _rand(a, b):
    return broadcast(And, b, a)

def _neg(a):
    return broadcast(USub, a)

def _invert(a):
    return broadcast(Not, a)


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
                    "Please specify schema keyword in .map method.\n"
                    "t['columnname'].map(function, schema='{col: type}')")

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


from .core import dshape_method_list, dshape_methods, method_properties

schema_method_list = [
    ]

schema_methods = memoize(partial(select_functions, schema_method_list))

dshape_method_list.extend([
    (iscollection, {shape, ndim, _ne, _lt, _le, _gt, _ge, _add, _radd, _mul,
        _rmul, _div, _rdiv, _floordiv, _rfloordiv, _sub, _rsub, _pow, _rpow,
        _mod, _rmod, _or, _ror, _and, _rand, _neg, _invert})
    ])

method_properties.update([shape, ndim])
