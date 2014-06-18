""" An abstract Table

>>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
>>> deadbeats = accounts['name'][accounts['amount'] < 0]
"""
from __future__ import absolute_import, division, print_function

from datashape import dshape, var, DataShape, Record, isdimension
import datashape
import operator
from toolz import concat, partial, first, pipe, compose
from toolz.curried import filter
from . import scalar
from .core import Expr, Scalar
from .scalar import ScalarSymbol
from .scalar import *
from ..utils import unique
from ..compatibility import _strtypes, builtins


class TableExpr(Expr):
    """ Super class for all Table Expressions """
    @property
    def dshape(self):
        return datashape.var * self.schema

    @property
    def columns(self):
        return self.schema[0].names

    @property
    def dtype(self):
        ds = self.schema[-1]
        if isinstance(ds, Record):
            if len(ds.fields.values()) > 1:
                raise TypeError("`.dtype` not defined for multicolumn object. "
                                "Use `.schema` instead")
            else:
                return dshape(first(ds.fields.values()))
        else:
            return dshape(ds)

    def __getitem__(self, key):
        if isinstance(key, _strtypes):
            if key not in self.columns:
                raise ValueError("Mismatched Column: %s" % str(key))
            return Column(self, key)
        if isinstance(key, list) and all(isinstance(k, _strtypes) for k in key):
            key = tuple(key)
            if not all(col in self.columns for col in key):
                raise ValueError("Mismatched Columns: %s" % str(key))
            return Projection(self, tuple(key))
        if isinstance(key, TableExpr):
            return Selection(self, key)
        raise TypeError("Did not understand input: %s[%s]" % (self, key))

    def sort(self, key=None, ascending=True):
        """ Sort table

        Parameters
        ----------
        key: string, list of strings, TableExpr
            Defines by what you want to sort.  Either:
                A single column string, ``t.sort('amount')``
                A list of column strings, ``t.sort(['name', 'amount'])``
                A Table Expression, ``t.sort(-t['amount'])``
        ascending: bool
            Determines order of the sort
        """
        if key is None:
            key = self.columns[0]
        return Sort(self, key, ascending)

    def head(self, n=10):
        return Head(self, n)

    def relabel(self, labels):
        return ReLabel(self, labels)

    def map(self, func, schema=None):
        return Map(self, func, schema)

    def count(self):
        return count(self)

    def distinct(self):
        return Distinct(self)

    def nunique(self):
        return nunique(self)

    def ancestors(self):
        return (self,)

    @property
    def iscolumn(self):
        if len(self.columns) > 1:
            return False
        raise NotImplementedError("%s.iscolumn not implemented" %
                str(type(self).__name__))



class TableSymbol(TableExpr):
    """ A Symbol for Tabular data

    This is a leaf in the expression tree

    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')
    >>> accounts['amount'] + 1
    accounts['amount'] + 1

    We define a TableSymbol with a name like ``accounts`` and the datashape of
    a single row, called a schema.
    """
    __slots__ = 'name', 'schema', 'iscolumn'

    def __init__(self, name, schema, iscolumn=False):
        self.name = name
        self.schema = dshape(schema)
        self.iscolumn = iscolumn

    def __str__(self):
        return self.name

    def ancestors(self):
        return (self,)


class RowWise(TableExpr):
    def ancestors(self):
        return (self,) + self.parent.ancestors()

class Projection(RowWise):
    """ Select columns from table

    SELECT a, b, c
    FROM table

    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')
    >>> accounts[['name', 'amount']].schema
    dshape("{ name : string, amount : int32 }")
    """
    __slots__ = 'parent', '_columns'

    def __init__(self, table, columns):
        self.parent = table
        self._columns = tuple(columns)

    @property
    def columns(self):
        return self._columns

    @property
    def schema(self):
        d = self.parent.schema[0].fields
        return DataShape(Record([(col, d[col]) for col in self.columns]))

    def __str__(self):
        return '%s[[%s]]' % (self.parent,
                             ', '.join(["'%s'" % col for col in self.columns]))

    @property
    def iscolumn(self):
        return False


class ColumnSyntaxMixin(object):
    def __eq__(self, other):
        return columnwise(Eq, self, other)

    def __ne__(self, other):
        return columnwise(NE, self, other)

    def __lt__(self, other):
        return columnwise(LT, self, other)

    def __le__(self, other):
        return columnwise(LE, self, other)

    def __gt__(self, other):
        return columnwise(GT, self, other)

    def __ge__(self, other):
        return columnwise(GE, self, other)

    def __add__(self, other):
        return columnwise(Add, self, other)

    def __radd__(self, other):
        return columnwise(Add, other, self)

    def __mul__(self, other):
        return columnwise(Mul, self, other)

    def __rmul__(self, other):
        return columnwise(Mul, other, self)

    def __div__(self, other):
        return columnwise(Div, self, other)

    __truediv__ = __div__

    def __rdiv__(self, other):
        return columnwise(Div, other, self)

    def __sub_(self, other):
        return columnwise(Sub, self, other)

    def __rsub__(self, other):
        return columnwise(Sub, other, self)

    def __pow__(self, other):
        return columnwise(Pow, self, other)

    def __rpow__(self, other):
        return columnwise(Pow, other, self)

    def __mod__(self, other):
        return columnwise(Mod, self, other)

    def __rmod__(self, other):
        return columnwise(Mod, other, self)

    def __neg__(self):
        return columnwise(Neg, self)

    def label(self, label):
        return Label(self, label)

    def sum(self):
        return sum(self)

    def min(self):
        return min(self)

    def max(self):
        return max(self)

    def any(self):
        return any(self)

    def all(self):
        return all(self)

    def mean(self):
        return mean(self)

    def var(self):
        return var(self)

    def std(self):
        return std(self)

    iscolumn = True


class Column(ColumnSyntaxMixin, Projection):
    """ A single column from a table

    SELECT a
    FROM table

    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')
    >>> accounts['name'].schema
    dshape("{ name : string }")
    """
    __slots__ = 'parent', 'column'

    __hash__ = Expr.__hash__

    iscolumn = True

    def __init__(self, table, column):
        self.parent = table
        self.column = column

    @property
    def columns(self):
        return (self.column,)

    def __str__(self):
        return "%s['%s']" % (self.parent, self.columns[0])

    @property
    def scalar_symbol(self):
        return ScalarSymbol(self.column, dtype=self.dtype)


class Selection(TableExpr):
    """ Filter rows of table based on predicate

    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')
    >>> deadbeats = accounts[accounts['amount'] < 0]
    """
    __slots__ = 'parent', 'predicate'

    def __init__(self, table, predicate):
        if predicate.dtype != dshape('bool'):
            raise TypeError("Must select over a boolean predicate.  Got:\n"
                            "%s[%s]" % (table, predicate))
        self.parent = table
        self.predicate = predicate  # A Relational

    def __str__(self):
        return "%s[%s]" % (self.parent, self.predicate)

    @property
    def schema(self):
        return self.parent.schema

    @property
    def iscolumn(self):
        return self.parent.iscolumn


def columnwise(op, *column_inputs):
    """ Merge columns with scalar operation


    Parameters
    ----------
    op - Scalar Operation like Add, Mul, Sin, Exp
    column_inputs - either Column, ColumnWise or constant (like 1, 1.0, '1')

    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')

    >>> columnwise(Add, accounts['amount'], 100)
    accounts['amount'] + 100

    Fuses operations down into ScalarExpr level

    >>> columnwise(Mul, 2, (accounts['amount'] + 100))
    2 * (accounts['amount'] + 100)
    """
    expr_inputs = []
    parents = set()
    for col in column_inputs:
        if isinstance(col, ColumnWise):
            expr_inputs.append(col.expr)
            parents.add(col.parent)
        elif isinstance(col, Column):
            # TODO: specify dtype
            expr_inputs.append(col.scalar_symbol)
            parents.add(col.parent)
        else:
            # maybe something like 5 or 'Alice'
            expr_inputs.append(col)

    if not len(parents) == 1:
        raise ValueError("All inputs must be from same Table.\n"
                         "Saw the following tables: %s"
                         % ', '.join(map(str, parents)))

    expr = op(*expr_inputs)
    return ColumnWise(first(parents), expr)


class ColumnWise(RowWise, ColumnSyntaxMixin):
    """ Apply Scalar Expression onto columns of data

    Parameters
    ----------

    parent - TableExpr
    expr - ScalarExpr
        The names of the varibles within the scalar expr must match the columns
        of the parent.  Use ``Column.scalar_variable`` to generate the
        appropriate ScalarSymbol

    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')

    >>> expr = Add(accounts['amount'].scalar_symbol, 100)
    >>> ColumnWise(accounts, expr)
    accounts['amount'] + 100
    """
    __slots__ = 'parent', 'expr'
    def __init__(self, parent, expr):
        self.parent = parent
        self.expr = expr

    __hash__ = Expr.__hash__

    iscolumn = True

    @property
    def schema(self):
        return self.expr.dshape

    def __str__(self):
        columns = self.active_columns()
        newcol = lambda c: "%s['%s']" % (self.parent, c)
        return eval_str(self.expr.subs(dict(zip(columns,
                                                map(newcol, columns)))))

    def active_columns(self):
        return sorted(unique(x.name for x in self.traverse()
                                    if isinstance(x, ScalarSymbol)))


class Join(TableExpr):
    """ Join two tables on common columns

    Parameters
    ----------
    lhs : TableExpr
    rhs : TableExpr
    on_left : string
    on_right : string

    >>> names = TableSymbol('names', '{name: string, id: int}')
    >>> amounts = TableSymbol('amounts', '{amount: int, id: int}')

    Join tables based on shared column name
    >>> joined = Join(names, amounts, 'id')

    Join based on different column names
    >>> amounts = TableSymbol('amounts', '{amount: int, acctNumber: int}')
    >>> joined = Join(names, amounts, 'id', 'acctNumber')
    """
    __slots__ = 'lhs', 'rhs', 'on_left', 'on_right'

    iscolumn = False

    def __init__(self, lhs, rhs, on_left, on_right=None):
        self.lhs = lhs
        self.rhs = rhs
        if not on_right:
            on_right = on_left
        self.on_left = on_left
        self.on_right = on_right
        if lhs.schema[0][on_left] != rhs.schema[0][on_right]:
            raise TypeError("Schema's of joining columns do not match")

    @property
    def schema(self):
        rec1 = self.lhs.schema[0]
        rec2 = self.rhs.schema[0]

        rec = rec1.parameters[0] + tuple((k, v) for k, v in rec2.parameters[0]
                                                 if  k != self.on_right)
        return dshape(Record(rec))


sqrt = partial(columnwise, scalar.sqrt)

sin = partial(columnwise, scalar.sin)
cos = partial(columnwise, scalar.cos)
tan = partial(columnwise, scalar.tan)
sinh = partial(columnwise, scalar.sinh)
cosh = partial(columnwise, scalar.cosh)
tanh = partial(columnwise, scalar.tanh)
acos = partial(columnwise, scalar.acos)
acosh = partial(columnwise, scalar.acosh)
asin = partial(columnwise, scalar.asin)
asinh = partial(columnwise, scalar.asinh)
atan = partial(columnwise, scalar.atan)
atanh = partial(columnwise, scalar.atanh)

exp = partial(columnwise, scalar.exp)
log = partial(columnwise, scalar.log)
expm1 = partial(columnwise, scalar.expm1)
log10 = partial(columnwise, scalar.log10)
log1p = partial(columnwise, scalar.log1p)

radians = partial(columnwise, scalar.radians)
degrees = partial(columnwise, scalar.degrees)

ceil = partial(columnwise, scalar.ceil)
floor = partial(columnwise, scalar.floor)
trunc = partial(columnwise, scalar.trunc)

isnan = partial(columnwise, scalar.isnan)


class Reduction(Scalar):
    """ A column-wise reduction

    >>> t = TableSymbol('t', '{name: string, amount: int, id: int}')
    >>> e = t['amount'].sum()

    >>> data = [['Alice', 100, 1],
    ...         ['Bob', 200, 2],
    ...         ['Alice', 50, 3]]

    >>> from blaze.compute.python import compute
    >>> compute(e, data)
    350
    """
    __slots__ = 'parent',

    def __init__(self, table):
        self.parent = table

    @property
    def dshape(self):
        return self.parent.dshape.subarray(1)

    @property
    def symbol(self):
        return type(self).__name__


class any(Reduction): pass
class all(Reduction): pass
class sum(Reduction): pass
class max(Reduction): pass
class min(Reduction): pass
class mean(Reduction): pass
class var(Reduction): pass
class std(Reduction): pass
class count(Reduction): pass
class nunique(Reduction): pass


class By(TableExpr):
    """ Split-Apply-Combine Operator

    >>> t = TableSymbol('t', '{name: string, amount: int, id: int}')
    >>> e = By(t, t['name'], t['amount'].sum())

    >>> data = [['Alice', 100, 1],
    ...         ['Bob', 200, 2],
    ...         ['Alice', 50, 3]]

    >>> from blaze.compute.python import compute
    >>> compute(e, data) #doctest: +SKIP
    {'Alice': 150, 'Bob': 200}
    """

    __slots__ = 'parent', 'grouper', 'apply'

    iscolumn = False

    def __init__(self, parent, grouper, apply):
        self.parent = parent
        s = TableSymbol('', parent.schema, parent.iscolumn)
        self.grouper = grouper.subs({parent: s})
        self.apply = apply.subs({parent: s})
        if isdimension(self.apply.dshape[0]):
            raise TypeError("Expected Reduction")

    @property
    def schema(self):
        group = self.grouper.schema[0].parameters[0]
        if isinstance(self.apply.dshape[0], Record):
            apply = self.apply.dshape[0].parameters[0]
        else:
            apply = (('0', self.apply.dshape),)

        params = unique(group + apply, key=lambda x: x[0])

        return dshape(Record(list(params)))


class Sort(TableExpr):
    """ Table in sorted order

    >>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
    >>> accounts.sort('amount', ascending=False).schema
    dshape("{ name : string, amount : int32 }")


    Some backends support sorting by arbitrary rowwise tables, e.g.

    >>> accounts.sort(-accounts['amount']) # doctest: +SKIP
    """
    __slots__ = 'parent', 'column', 'ascending'

    def __init__(self, parent, column, ascending=True):
        self.parent = parent
        self.column = column
        self.ascending = ascending

    @property
    def schema(self):
        return self.parent.schema

    @property
    def iscolumn(self):
        return self.parent.iscolumn


class Distinct(TableExpr):
    """ Distinct elements filter

    >>> t = TableSymbol('t', '{name: string, amount: int, id: int}')
    >>> e = Distinct(t)

    >>> data = [('Alice', 100, 1),
    ...         ('Bob', 200, 2),
    ...         ('Alice', 100, 1)]

    >>> from blaze.compute.python import compute
    >>> sorted(compute(e, data))
    [('Alice', 100, 1), ('Bob', 200, 2)]
    """
    __slots__ = 'parent',

    def __init__(self, table):
        self.parent = table

    @property
    def schema(self):
        return self.parent.schema

    @property
    def iscolumn(self):
        return self.parent.iscolumn

class Head(TableExpr):
    """ First ``n`` elements of table

    >>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
    >>> accounts.head(5).dshape
    dshape("5 * { name : string, amount : int32 }")
    """
    __slots__ = 'parent', 'n'

    def __init__(self, parent, n=10):
        self.parent = parent
        self.n = n

    @property
    def schema(self):
        return self.parent.schema

    @property
    def dshape(self):
        return self.n * self.schema

    @property
    def iscolumn(self):
        return self.parent.iscolumn


class Label(RowWise, ColumnSyntaxMixin):
    """ A Labeled column

    >>> accounts = TableSymbol('accounts', '{name: string, amount: int}')

    >>> (accounts['amount'] * 100).schema
    dshape("float64")

    >>> (accounts['amount'] * 100).label('new_amount').schema #doctest: +SKIP
    dshape("{ new_amount : float64 }")
    """
    __slots__ = 'parent', 'label'

    def __init__(self, parent, label):
        self.parent = parent
        self.label = label

    @property
    def schema(self):
        if isinstance(self.parent.schema[0], Record):
            dtype = self.parent.schema[0].fields.values()[0]
        else:
            dtype = self.parent.schema[0]
        return DataShape(Record([[self.label, dtype]]))


class ReLabel(RowWise):
    """ Table with same content but new labels

    >>> accounts = TableSymbol('accounts', '{name: string, amount: int}')

    >>> accounts.schema
    dshape("{ name : string, amount : int32 }")
    >>> accounts.relabel({'amount': 'balance'}).schema
    dshape("{ name : string, balance : int32 }")
    """
    __slots__ = 'parent', 'labels'

    def __init__(self, parent, labels):
        self.parent = parent
        if isinstance(labels, dict):  # Turn dict into tuples
            labels = tuple(sorted(labels.items()))
        self.labels = labels

    @property
    def schema(self):
        subs = dict(self.labels)
        d = self.parent.schema[0].fields

        return DataShape(Record([[subs.get(name, name), dtype]
            for name, dtype in self.parent.schema[0].parameters[0]]))

    @property
    def iscolumn(self):
        return self.parent.iscolumn


class Map(RowWise):
    """ Map an arbitrary Python function across rows in a Table

    >>> from datetime import datetime

    >>> t = TableSymbol('t', '{price: real, time: int64}')  # times as integers
    >>> datetimes = t['time'].map(datetime.utcfromtimestamp)

    Optionally provide extra schema information

    >>> datetimes = t['time'].map(datetime.utcfromtimestamp,
    ...                           schema='{time: datetime}')

    See Also:
        Apply
    """
    __slots__ = 'parent', 'func', '_schema'

    def __init__(self, parent, func, schema=None):
        self.parent = parent
        self.func = func
        self._schema = schema

    @property
    def schema(self):
        if self._schema:
            return dshape(self._schema)
        else:
            raise NotImplementedError()


class Apply(TableExpr):
    """ Apply an arbitrary Python function onto a Table

    >>> t = TableSymbol('t', '{name: string, amount: int}')
    >>> h = Apply(hash, t)  # Hash value of resultant table

    Optionally provide extra datashape information

    >>> h = Apply(hash, t, dshape='real')

    Apply brings a function within the expression tree.
    The following transformation is often valid

    Before ``compute(Apply(f, expr), ...)``
    After  ``f(compute(expr, ...)``

    See Also:
        Map
    """
    __slots__ = 'parent', 'func', '_dshape'

    def __init__(self, func, parent, dshape=None):
        self.parent = parent
        self.func = func
        self._dshape = dshape

    @property
    def schema(self):
        if isdimension(self.dshape[0]):
            return self.dshape.subshape[0]
        else:
            return TypeError("Non-tabular datashape, %s" % self.dshape)

    @property
    def dshape(self):
        if self._dshape:
            return dshape(self._dshape)
        else:
            return NotImplementedError("Datashape of arbitrary Apply not defined")


def common_ancestor(*tables):
    """ Common ancestor between subtables

    >>> t = TableSymbol('t', '{x: int, y: int}')
    >>> common_ancestor(t['x'], t['y'])
    t
    """
    sets = [set(t.ancestors()) for t in tables]
    return builtins.max(set.intersection(*sets),
                        key=compose(len, str))

def merge(*tables):
    # Get common ancestor
    parent = common_ancestor(*tables)
    if not parent:
        raise ValueError("No common ancestor found for input tables")

    shim = TableSymbol('_ancestor', parent.schema, parent.iscolumn)

    tables = tuple(t.subs({parent: shim}) for t in tables)
    return Merge(parent, tables)


class Merge(RowWise):
    """ Merge many Tables together

    Must all descend from same table via RowWise operations

    >>> accounts = TableSymbol('accounts', '{name: string, amount: int}')

    >>> newamount = (accounts['amount'] * 1.5).label('new_amount')

    >>> merge(accounts, newamount).columns
    ['name', 'amount', 'new_amount']
    """
    __slots__ = 'parent', 'children'

    iscolumn = False

    def __init__(self, parent, children):
        # TODO: Assert all parents descend from the same parent via RowWise
        # operations
        self.parent = parent
        self.children = children

    @property
    def schema(self):
        for c in self.children:
            if not isinstance(c.schema[0], Record):
                raise TypeError("All schemas must have Record shape.  Got %s" %
                                c.schema[0])
        return dshape(Record(list(concat(c.schema[0].parameters[0] for c in
            self.children))))
