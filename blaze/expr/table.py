""" An abstract Table

>>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
>>> deadbeats = accounts['name'][accounts['amount'] < 0]
"""
from __future__ import absolute_import, division, print_function

from abc import abstractproperty
from datashape import dshape, var, DataShape, Record, isdimension, Option
from datashape import coretypes as ct
import datashape
from toolz import concat, partial, first, compose, get, unique
from . import scalar
from .core import Expr, path
from .scalar import ScalarSymbol
from .scalar import (Eq, Ne, Lt, Le, Gt, Ge, Add, Mult, Div, Sub, Pow, Mod, Or,
                     And, USub, Not, eval_str, Scalar, FloorDiv)
from ..compatibility import _strtypes, builtins


class TableExpr(Expr):
    """ Super class for all Table Expressions

    This is not intended to be constructed by users.

    See Also
    --------

    blaze.expr.table.TableSymbol
    """
    __inputs__ = 'child',

    @property
    def dshape(self):
        return datashape.var * self.schema

    @property
    def columns(self):
        if isinstance(self.schema[0], Record):
            return self.schema[0].names

    @abstractproperty
    def schema(self):
        pass

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

    def __getitem__(self, key):
        if isinstance(key, _strtypes):
            if key not in self.columns:
                raise ValueError("Mismatched Column: %s" % str(key))
            return Column(self, key)
        if isinstance(key, list) and builtins.all(isinstance(k, _strtypes) for k in key):
            if not builtins.all(col in self.columns for col in key):
                raise ValueError("Mismatched Columns: %s" % str(key))
            return projection(self, key)
        if isinstance(key, TableExpr):
            return selection(self, key)
        raise TypeError("Did not understand input: %s[%s]" % (self, key))

    def __getattr__(self, key):
        if key in self.columns:
            return self[key]
        return object.__getattribute__(self, key)

    def __dir__(self):
        return sorted(set(dir(type(self)) + list(self.__dict__.keys()) +
                          self.columns))

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
        return sort(self, key, ascending)

    def head(self, n=10):
        return head(self, n)

    def relabel(self, labels):
        return relabel(self, labels)

    def map(self, func, schema=None, iscolumn=None):
        return Map(self, func, schema, iscolumn)

    def count(self):
        return count(self)

    def distinct(self):
        return distinct(self)

    def nunique(self):
        return nunique(self)

    def min(self):
        return min(self)

    def max(self):
        return max(self)

    @property
    def iscolumn(self):
        if len(self.columns) > 1:
            return False
        raise NotImplementedError("%s.iscolumn not implemented" %
                str(type(self).__name__))


class TableSymbol(TableExpr):
    """ A Symbol for Tabular data

    This is a leaf in the expression tree

    Examples
    --------

    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')
    >>> accounts['amount'] + 1
    accounts['amount'] + 1

    We define a TableSymbol with a name like ``accounts`` and the datashape of
    a single row, called a schema.
    """
    __slots__ = '_name', 'schema', 'iscolumn'
    __inputs__ = ()

    def __init__(self, name, schema, iscolumn=False):
        self._name = name
        if isinstance(schema, _strtypes):
            schema = dshape(schema)
        self.schema = schema
        self.iscolumn = iscolumn

    def __str__(self):
        return self._name

    def resources(self):
        return dict()


class RowWise(TableExpr):
    """ Apply an operation equally to each of the rows.  An Interface.

    Common rowwise operations include ``Map``, ``ColumnWise``, ``Projection``,
    and anything else that operates by applying some transformation evenly
    across all rows in a table.

    RowWise operations have the same number of rows as their children

    See Also
    --------

    blaze.expr.table.Projection
    blaze.expr.table.Map
    blaze.expr.table.ColumnWise
    blaze.expr.table.
    blaze.expr.table.
    blaze.expr.table.
    """


class Projection(RowWise):
    """ Select columns from table

    SELECT a, b, c
    FROM table

    Examples
    --------

    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')
    >>> accounts[['name', 'amount']].schema
    dshape("{ name : string, amount : int32 }")

    See Also
    --------

    blaze.expr.table.Column
    """
    __slots__ = 'child', '_columns'

    @property
    def columns(self):
        return list(self._columns)

    @property
    def schema(self):
        d = self.child.schema[0].dict
        return DataShape(Record([(col, d[col]) for col in self.columns]))

    def __str__(self):
        return '%s[[%s]]' % (self.child,
                             ', '.join(["'%s'" % col for col in self.columns]))

    @property
    def iscolumn(self):
        return False


def projection(table, columns):
    return Projection(table, tuple(columns))

projection.__doc__ = Projection.__doc__


class ColumnSyntaxMixin(object):
    """ Syntax bits for table expressions of column shape """
    iscolumn = True

    def __eq__(self, other):
        return columnwise(Eq, self, other)

    def __ne__(self, other):
        return columnwise(Ne, self, other)

    def __lt__(self, other):
        return columnwise(Lt, self, other)

    def __le__(self, other):
        return columnwise(Le, self, other)

    def __gt__(self, other):
        return columnwise(Gt, self, other)

    def __ge__(self, other):
        return columnwise(Ge, self, other)

    def __add__(self, other):
        return columnwise(Add, self, other)

    def __radd__(self, other):
        return columnwise(Add, other, self)

    def __mul__(self, other):
        return columnwise(Mult, self, other)

    def __rmul__(self, other):
        return columnwise(Mult, other, self)

    def __div__(self, other):
        return columnwise(Div, self, other)

    __truediv__ = __div__
    __rtruediv__ = __truediv__

    def __floordiv__(self, other):
        return columnwise(FloorDiv, self, other)

    def __rfloordiv__(self, other):
        return columnwise(FloorDiv, other, self)

    def __rdiv__(self, other):
        return columnwise(Div, other, self)

    def __sub__(self, other):
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

    def __or__(self, other):
        return columnwise(Or, self, other)

    def __ror__(self, other):
        return columnwise(Or, other, self)

    def __and__(self, other):
        return columnwise(And, self, other)

    def __rand__(self, other):
        return columnwise(And, other, self)

    def __neg__(self):
        return columnwise(USub, self)

    def __invert__(self):
        return columnwise(Not, self)

    def label(self, label):
        return Label(self, label)

    def sum(self):
        return sum(self)

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

    def isnan(self):
        return columnwise(scalar.isnan, self)


class Column(ColumnSyntaxMixin, Projection):
    """ A single column from a table

    SELECT a
    FROM table

    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')
    >>> accounts['name'].schema
    dshape("{ name : string }")

    See Also
    --------

    blaze.expr.table.Projection
    """
    __slots__ = 'child', 'column'

    __hash__ = Expr.__hash__

    iscolumn = True

    @property
    def columns(self):
        return [self.column]

    def __str__(self):
        return "%s['%s']" % (self.child, self.columns[0])

    @property
    def scalar_symbol(self):
        return ScalarSymbol(self.column, dtype=self.dtype)


class Selection(TableExpr):
    """ Filter rows of table based on predicate

    Examples
    --------

    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')
    >>> deadbeats = accounts[accounts['amount'] < 0]
    """
    __slots__ = 'child', 'predicate'

    def __str__(self):
        return "%s[%s]" % (self.child, self.predicate)

    @property
    def schema(self):
        return self.child.schema

    @property
    def iscolumn(self):
        return self.child.iscolumn


def selection(table, predicate):
    subexpr = common_subexpression(table, predicate)

    if not builtins.all(isinstance(node, (RowWise, TableSymbol))
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


def _expr_child(col):
    """ Expr and child of column

    Examples
    --------

    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')
    >>> _expr_child(accounts['name'])
    (name, accounts)

    Helper function for ``columnwise``
    """
    if isinstance(col, ColumnWise):
        return col.expr, col.child
    elif isinstance(col, Column):
        # TODO: specify dtype
        return col.scalar_symbol, col.child
    elif isinstance(col, Label):
        return _expr_child(col.child)
    else:
        return col, None


def columnwise(op, *column_inputs):
    """ Merge columns with scalar operation


    Parameters
    ----------
    op : Scalar Operation like Add, Mult, Sin, Exp

    column_inputs : either Column, ColumnWise or constant (like 1, 1.0, '1')

    Examples
    --------

    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')

    >>> columnwise(Add, accounts['amount'], 100)
    accounts['amount'] + 100

    Fuses operations down into ScalarExpr level

    >>> columnwise(Mult, 2, (accounts['amount'] + 100))
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

    return ColumnWise(first(children), expr)


class ColumnWise(RowWise, ColumnSyntaxMixin):
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

    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')

    >>> expr = Add(accounts['amount'].scalar_symbol, 100)
    >>> ColumnWise(accounts, expr)
    accounts['amount'] + 100

    See Also
    --------

    blaze.expr.table.columnwise
    """
    __slots__ = 'child', 'expr'

    __hash__ = Expr.__hash__

    iscolumn = True

    @property
    def schema(self):
        return self.expr.dshape

    def __str__(self):
        columns = self.active_columns()
        newcol = lambda c: "%s['%s']" % (self.child, c)
        return eval_str(self.expr.subs(dict(zip(columns,
                                                map(newcol, columns)))))

    def active_columns(self):
        return sorted(unique(x._name for x in self.traverse()
                                    if isinstance(x, ScalarSymbol)))


def unpack(l):
    if isinstance(l, (tuple, list, set)) and len(l) == 1:
        return next(iter(l))
    else:
        return l


class Join(TableExpr):
    """ Join two tables on common columns

    Parameters
    ----------
    lhs : TableExpr
    rhs : TableExpr
    on_left : string
    on_right : string

    Examples
    --------

    >>> names = TableSymbol('names', '{name: string, id: int}')
    >>> amounts = TableSymbol('amounts', '{amount: int, id: int}')

    Join tables based on shared column name
    >>> joined = join(names, amounts, 'id')

    Join based on different column names
    >>> amounts = TableSymbol('amounts', '{amount: int, acctNumber: int}')
    >>> joined = join(names, amounts, 'id', 'acctNumber')

    See Also
    --------

    blaze.expr.table.Merge
    blaze.expr.table.Union
    """
    __slots__ = 'lhs', 'rhs', '_on_left', '_on_right', 'how'
    __inputs__ = 'lhs', 'rhs'

    iscolumn = False

    @property
    def on_left(self):
        if isinstance(self._on_left, tuple):
            return list(self._on_left)
        else:
            return self._on_left

    @property
    def on_right(self):
        if isinstance(self._on_left, tuple):
            return list(self._on_left)
        else:
            return self._on_left

    @property
    def schema(self):
        """

        Examples
        --------

        >>> t = TableSymbol('t', '{name: string, amount: int}')
        >>> s = TableSymbol('t', '{name: string, id: int}')

        >>> join(t, s).schema
        dshape("{ name : string, amount : int32, id : int32 }")

        >>> join(t, s, how='left').schema
        dshape("{ name : string, amount : int32, id : ?int32 }")
        """
        option = lambda dt: dt if isinstance(dt, Option) else Option(dt)

        joined = [[name, dt] for name, dt in self.lhs.schema[0].parameters[0]
                        if name in self.on_left]

        left = [[name, dt] for name, dt in self.lhs.schema[0].parameters[0]
                           if name not in self.on_left]

        right = [[name, dt] for name, dt in self.rhs.schema[0].parameters[0]
                            if name not in self.on_right]

        if self.how in ('right', 'outer'):
            left = [[name, option(dt)] for name, dt in left]
        if self.how in ('left', 'outer'):
            right = [[name, option(dt)] for name, dt in right]

        return dshape(Record(joined + left + right))


def join(lhs, rhs, on_left=None, on_right=None, how='inner'):
    if not on_left and not on_right:
        on_left = on_right = unpack(list(sorted(
            set(lhs.columns) & set(rhs.columns),
            key=lhs.columns.index)))
    if not on_right:
        on_right = on_left
    if isinstance(on_left, tuple):
        on_left = list(on_left)
    if isinstance(on_right, tuple):
        on_right = list(on_right)
    if get(on_left, lhs.schema[0]) != get(on_right, rhs.schema[0]):
        raise TypeError("Schema's of joining columns do not match")
    _on_left = tuple(on_left) if isinstance(on_left, list) else on_left
    _on_right = (tuple(on_right) if isinstance(on_right, list)
                        else on_right)

    how = how.lower()
    if how not in ('inner', 'outer', 'left', 'right'):
        raise ValueError("How parameter should be one of "
                         "\n\tinner, outer, left, right."
                         "\nGot: %s" % how)

    return Join(lhs, rhs, _on_left, _on_right, how)


join.__doc__ = Join.__doc__


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

    Blaze supports the same class of reductions as NumPy and Pandas.

        sum, min, max, any, all, mean, var, std, count, nunique

    Examples
    --------

    >>> t = TableSymbol('t', '{name: string, amount: int, id: int}')
    >>> e = t['amount'].sum()

    >>> data = [['Alice', 100, 1],
    ...         ['Bob', 200, 2],
    ...         ['Alice', 50, 3]]

    >>> from blaze.compute.python import compute
    >>> compute(e, data)
    350
    """
    __slots__ = 'child',
    dtype = None

    @property
    def dshape(self):
        if self.child.columns and len(self.child.columns) == 1:
            name = self.child.columns[0] + '_' + type(self).__name__
            return DataShape(Record([[name, self.dtype]]))
        else:
            return DataShape(Record([[type(self).__name__, self.dtype]]))

    @property
    def symbol(self):
        return type(self).__name__


class any(Reduction):
    dtype = ct.bool_
class all(Reduction):
    dtype = ct.bool_
class sum(Reduction):
    @property
    def dtype(self):
        schema = self.child.schema[0]
        if isinstance(schema, Record) and len(schema.types) == 1:
            return first(schema.types)
        else:
            return schema
class max(Reduction):
    @property
    def dtype(self):
        schema = self.child.schema[0]
        if isinstance(schema, Record) and len(schema.types) == 1:
            return first(schema.types)
        else:
            return schema
class min(Reduction):
    @property
    def dtype(self):
        schema = self.child.schema[0]
        if isinstance(schema, Record) and len(schema.types) == 1:
            return first(schema.types)
        else:
            return schema
class mean(Reduction):
    dtype = ct.real
class var(Reduction):
    dtype = ct.real
class std(Reduction):
    dtype = ct.real
class count(Reduction):
    dtype = ct.int_
class nunique(Reduction):
    dtype = ct.int_


class By(TableExpr):
    """ Split-Apply-Combine Operator

    Examples
    --------

    >>> t = TableSymbol('t', '{name: string, amount: int, id: int}')
    >>> e = by(t, t['name'], t['amount'].sum())

    >>> data = [['Alice', 100, 1],
    ...         ['Bob', 200, 2],
    ...         ['Alice', 50, 3]]

    >>> from blaze.compute.python import compute
    >>> sorted(compute(e, data))
    [('Alice', 150), ('Bob', 200)]
    """

    __slots__ = 'child', 'grouper', 'apply'

    iscolumn = False

    @property
    def schema(self):
        group = self.grouper.schema[0].parameters[0]
        reduction_name = type(self.apply).__name__
        if isinstance(self.apply.dshape[0], Record):
            apply = self.apply.dshape[0].parameters[0]
        else:
            apply = ((reduction_name, self.apply.dshape),)

        params = unique(group + apply, key=lambda x: x[0])

        return dshape(Record(list(params)))


def by(child, grouper, apply):
    if isdimension(apply.dshape[0]):
        raise TypeError("Expected Reduction")
    return By(child, grouper, apply)

by.__doc__ = By.__doc__


class Sort(TableExpr):
    """ Table in sorted order

    Examples
    --------

    >>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
    >>> accounts.sort('amount', ascending=False).schema
    dshape("{ name : string, amount : int32 }")

    Some backends support sorting by arbitrary rowwise tables, e.g.

    >>> accounts.sort(-accounts['amount']) # doctest: +SKIP
    """
    __slots__ = 'child', '_key', 'ascending'

    @property
    def schema(self):
        return self.child.schema

    @property
    def iscolumn(self):
        return self.child.iscolumn

    @property
    def key(self):
        if isinstance(self._key, tuple):
            return list(self._key)
        else:
            return self._key


def sort(child, key, ascending=True):
    if isinstance(key, list):
        key = tuple(key)
    return Sort(child, key, ascending)


class Distinct(TableExpr):
    """
    Removes duplicate rows from the table, so every row is distinct

    Examples
    --------

    >>> t = TableSymbol('t', '{name: string, amount: int, id: int}')
    >>> e = distinct(t)

    >>> data = [('Alice', 100, 1),
    ...         ('Bob', 200, 2),
    ...         ('Alice', 100, 1)]

    >>> from blaze.compute.python import compute
    >>> sorted(compute(e, data))
    [('Alice', 100, 1), ('Bob', 200, 2)]
    """
    __slots__ = 'child',

    @property
    def schema(self):
        return self.child.schema

    @property
    def iscolumn(self):
        return self.child.iscolumn


distinct = Distinct


class Head(TableExpr):
    """ First ``n`` elements of table

    Examples
    --------

    >>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
    >>> accounts.head(5).dshape
    dshape("5 * { name : string, amount : int32 }")
    """
    __slots__ = 'child', 'n'

    @property
    def schema(self):
        return self.child.schema

    @property
    def dshape(self):
        return self.n * self.schema

    @property
    def iscolumn(self):
        return self.child.iscolumn


def head(child, n=10):
    return Head(child, n)

head.__doc__ = Head.__doc__


class Label(RowWise, ColumnSyntaxMixin):
    """ A Labeled column

    Examples
    --------

    >>> accounts = TableSymbol('accounts', '{name: string, amount: int}')

    >>> (accounts['amount'] * 100).schema
    dshape("float64")

    >>> (accounts['amount'] * 100).label('new_amount').schema #doctest: +SKIP
    dshape("{ new_amount : float64 }")

    See Also
    --------

    blaze.expr.table.ReLabel
    """
    __slots__ = 'child', 'label'

    @property
    def schema(self):
        if isinstance(self.child.schema[0], Record):
            dtype = self.child.schema[0].types[0]
        else:
            dtype = self.child.schema[0]
        return DataShape(Record([[self.label, dtype]]))


class ReLabel(RowWise):
    """
    Table with same content but with new labels

    Examples
    --------

    >>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
    >>> accounts.schema
    dshape("{ name : string, amount : int32 }")
    >>> accounts.relabel({'amount': 'balance'}).schema
    dshape("{ name : string, balance : int32 }")

    See Also
    --------

    blaze.expr.table.Label
    """
    __slots__ = 'child', 'labels'

    @property
    def schema(self):
        subs = dict(self.labels)
        d = self.child.schema[0].dict

        return DataShape(Record([[subs.get(name, name), dtype]
            for name, dtype in self.child.schema[0].parameters[0]]))

    @property
    def iscolumn(self):
        return self.child.iscolumn

def relabel(child, labels):
    if isinstance(labels, dict):  # Turn dict into tuples
        labels = tuple(sorted(labels.items()))
    return ReLabel(child, labels)

relabel.__doc__ = ReLabel.__doc__


class Map(RowWise):
    """ Map an arbitrary Python function across rows in a Table

    Examples
    --------

    >>> from datetime import datetime

    >>> t = TableSymbol('t', '{price: real, time: int64}')  # times as integers
    >>> datetimes = t['time'].map(datetime.utcfromtimestamp)

    Optionally provide extra schema information

    >>> datetimes = t['time'].map(datetime.utcfromtimestamp,
    ...                           schema='{time: datetime}')

    See Also
    --------

    blaze.expr.table.Apply
    """
    __slots__ = 'child', 'func', '_schema', '_iscolumn'

    @property
    def schema(self):
        if self._schema:
            return dshape(self._schema)
        else:
            raise NotImplementedError("Schema of mapped column not known.\n"
                    "Please specify schema keyword in .map method.\n"
                    "t['columnname'].map(function, schema='{col: type}')")

    @property
    def iscolumn(self):
        if self._iscolumn is not None:
            return self._iscolumn
        if self.child.iscolumn is not None:
            return self.child.iscolumn
        return self.schema[0].values()


class Apply(TableExpr):
    """ Apply an arbitrary Python function onto a Table

    Examples
    --------

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

    blaze.expr.table.Map
    """
    __slots__ = 'child', 'func', '_dshape'

    def __init__(self, func, child, dshape=None):
        self.child = child
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


def common_subexpression(*tables):
    """ Common sub expression between subtables

    Examples
    --------

    >>> t = TableSymbol('t', '{x: int, y: int}')
    >>> common_subexpression(t['x'], t['y'])
    t
    """
    sets = [set(t.subterms()) for t in tables]
    return builtins.max(set.intersection(*sets),
                        key=compose(len, str))

def merge(*tables):
    # Get common sub expression
    child = common_subexpression(*tables)
    if not child:
        raise ValueError("No common sub expression found for input tables")

    return Merge(child, tables)


class Merge(RowWise):
    """ Merge the columns of many Tables together

    Must all descend from same table via RowWise operations

    Examples
    --------

    >>> accounts = TableSymbol('accounts', '{name: string, amount: int}')

    >>> newamount = (accounts['amount'] * 1.5).label('new_amount')

    >>> merge(accounts, newamount).columns
    ['name', 'amount', 'new_amount']

    See Also
    --------

    blaze.expr.table.Union
    blaze.expr.table.Join
    """
    __slots__ = 'child', 'children'

    iscolumn = False

    @property
    def schema(self):
        for c in self.children:
            if not isinstance(c.schema[0], Record):
                raise TypeError("All schemas must have Record shape.  Got %s" %
                                c.schema[0])
        return dshape(Record(list(concat(c.schema[0].parameters[0] for c in
            self.children))))


class Union(TableExpr):
    """ Merge the rows of many Tables together

    Must all have the same schema

    Examples
    --------

    >>> usa_accounts = TableSymbol('accounts', '{name: string, amount: int}')
    >>> euro_accounts = TableSymbol('accounts', '{name: string, amount: int}')

    >>> all_accounts = union(usa_accounts, euro_accounts)
    >>> all_accounts.columns
    ['name', 'amount']

    See Also
    --------

    blaze.expr.table.Merge
    blaze.expr.table.Join
    """
    __slots__ = 'children',
    __inputs__ = 'children',

    def subterms(self):
        yield self
        for i in self.children:
            for node in i.subterms():
                yield node

    @property
    def schema(self):
        return self.children[0].schema


def union(*children):
    schemas = set(child.schema for child in children)
    if len(schemas) != 1:
        raise ValueError("Inconsistent schemas:\n\t%s" %
                            '\n\t'.join(map(str, schemas)))
    return Union(children)
