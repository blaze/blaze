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
from .predicates import iscolumn
from datashape.predicates import isunit
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
        if isinstance(key, _strtypes) and key in self.names:
            return self.get_field(key)
        if isinstance(key, Collection):
            return selection(self, key)
        if (isinstance(key, list)
                and builtins.all(isinstance(k, _strtypes) for k in key)):
            if set(key).issubset(self.names):
                return self.project(key)
            else:
                raise ValueError('Names %s not consistent with known names %s'
                        % (key, self.names))
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
                if isunit(self.dshape.measure): # t.foo.foo is t.foo
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
        if (isunit(self.dshape.measure) and
                (not isinstance(other, Expr)
                     or isunit(other.dshape.measure))):
            return broadcast(Eq, self, other)
        else:
            return self.isidentical(other)

    def __ne__(self, other):
        return broadcast(Ne, self, other)

    def __lt__(self, other):
        return broadcast(Lt, self, other)

    def __le__(self, other):
        return broadcast(Le, self, other)

    def __gt__(self, other):
        return broadcast(Gt, self, other)

    def __ge__(self, other):
        return broadcast(Ge, self, other)

    def __add__(self, other):
        return broadcast(Add, self, other)

    def __radd__(self, other):
        return broadcast(Add, other, self)

    def __mul__(self, other):
        return broadcast(Mult, self, other)

    def __rmul__(self, other):
        return broadcast(Mult, other, self)

    def __div__(self, other):
        return broadcast(Div, self, other)

    def __rdiv__(self, other):
        return broadcast(Div, other, self)

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    def __floordiv__(self, other):
        return broadcast(FloorDiv, self, other)

    def __rfloordiv__(self, other):
        return broadcast(FloorDiv, other, self)

    def __sub__(self, other):
        return broadcast(Sub, self, other)

    def __rsub__(self, other):
        return broadcast(Sub, other, self)

    def __pow__(self, other):
        return broadcast(Pow, self, other)

    def __rpow__(self, other):
        return broadcast(Pow, other, self)

    def __mod__(self, other):
        return broadcast(Mod, self, other)

    def __rmod__(self, other):
        return broadcast(Mod, other, self)

    def __or__(self, other):
        return broadcast(Or, self, other)

    def __ror__(self, other):
        return broadcast(Or, other, self)

    def __and__(self, other):
        return broadcast(And, self, other)

    def __rand__(self, other):
        return broadcast(And, other, self)

    def __neg__(self):
        return broadcast(USub, self)

    def __invert__(self):
        return broadcast(Not, self)

    def map(self, func, schema=None, name=None):
        return Map(self, func, schema, name)


class Symbol(Collection):
    """
    Symbolic data

    >>> points = Symbol('points', '5 * 3 * {x: int, y: int}')
    """
    __slots__ = '_name', 'dshape'

    def __init__(self, name, dshape):
        self._name = name
        if isinstance(dshape, _strtypes):
            dshape = datashape.dshape(dshape)
        self.dshape = dshape


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
        if isinstance(key, list) and set(key).issubset(set(self.names)):
            return self.child[key]
        raise ValueError("Column Mismatch: %s" % key)

    def get_field(self, fieldname):
        if fieldname in self.names:
            return Field(self.child, fieldname)
        raise ValueError("Field %s not found in columns %s" % (fieldname,
            self.names))


def projection(expr, names):
    if not isinstance(names, (tuple, list)):
        raise TypeError("Wanted list of strings, got %s" % names)
    if not set(names).issubset(expr.names):
        raise ValueError("Mismatched names.  Asking for names %s "
                "where expression has names %s" % (names, expr.names))
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

    @property
    def iscolumn(self):
        return self.child.iscolumn


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
    iscolumn = True
    __slots__ = 'child', 'label'

    @property
    def schema(self):
        return self.child.schema

    @property
    def _name(self):
        return self.label

    def get_field(self, key):
        if key[0] == self.names[0]:
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

    iscolumn = True

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
        assert iscolumn(self)
        return Map(self.child,
                   self.func,
                   self.schema,
                   name)

    @property
    def _name(self):
        if self._name0:
            return self._name0
        else:
            return self.child._name


def shape(expr):
    s = list(expr.dshape.shape)
    for i, elem in enumerate(s):
        try:
            s[i] = int(elem)
        except TypeError:
            pass

    return tuple(s)

from .core import dshape_method_list, dshape_methods, method_properties

schema_method_list = [
    ]

schema_methods = memoize(partial(select_functions, schema_method_list))

dshape_method_list.extend([
    (lambda ds: len(ds.shape), {shape})
    ])

method_properties.add(shape)
