""" An abstract Table

>>> accounts = TableSymbol('{name: string, amount: int}')
>>> deadbeats = accounts['name'][accounts['amount'] < 0]
"""

from __future__ import absolute_import, division, print_function

from datashape import dshape, var, DataShape, Record, isdimension
import datashape
import operator
from .core import Expr, Scalar, _str
from ..utils import unique
from ..compatibility import builtins


operators = '+-/*=|&!~<>'


class TableExpr(Expr):
    """ Super class for all Table Expressions """
    @property
    def dshape(self):
        return datashape.var * self.schema

    @property
    def columns(self):
        return self.schema[0].names

    def __getitem__(self, key):
        if isinstance(key, Boolean):
            return Selection(self, key)
        if isinstance(key, (tuple, list)):
            key = tuple(key)
            if not all(col in self.columns for col in key):
                raise ValueError("Mismatched Columns: %s" % str(key))
            return Projection(self, tuple(key))
        else:
            if key not in self.columns:
                raise ValueError("Mismatched Column: %s" % str(key))
            return Column(self, key)

    def sort(self, column=None, ascending=True):
        if column is None:
            column = self.columns[0]
        return Sort(self, column, ascending)

    def head(self, n=10):
        return Head(self, n)


class TableSymbol(TableExpr):
    """ A Symbol for Tabular data

    This is a leaf in the expression tree

    >>> t = TableSymbol('{name: string, amount: int, id: int}')
    """
    __slots__ = 'schema',

    def __init__(self, schema):
        self.schema = dshape(schema)

    def __str__(self):
        return type(self).__name__ + "('%s')" % self.schema


class Projection(TableExpr):
    """

    SELECT a, b, c
    FROM table
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
        return '%s[%s]' % (self.parent,
                           ', '.join(["'%s'" % col for col in self.columns]))


class Column(Projection):
    """

    SELECT a
    FROM table
    """
    __slots__ = 'parent', 'column'

    __hash__ = Expr.__hash__

    def __init__(self, table, column):
        self.parent = table
        self.column = column

    @property
    def columns(self):
        return (self.column,)

    def __str__(self):
        return "%s['%s']" % (self.parent, self.columns[0])

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __eq__(self, other):
        return Eq(self, other)

    def __lt__(self, other):
        return LT(self, other)

    def __gt__(self, other):
        return GT(self, other)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    def __div__(self, other):
        return Div(self, other)

    def __rdiv__(self, other):
        return Div(other, self)

    def __sub_(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __pow__(self, other):
        return Pow(self, other)

    def __rpow__(self, other):
        return Pow(other, self)

    def __mod__(self, other):
        return Mod(self, other)

    def __rmod__(self, other):
        return Mod(other, self)

    def count(self):
        return count(self)

    def nunique(self):
        return nunique(self)

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


class Selection(TableExpr):
    """
    WHERE a op b
    """
    __slots__ = 'parent', 'predicate'

    def __init__(self, table, predicate):
        self.parent = table
        self.predicate = predicate  # A Relational

    def __str__(self):
        return "%s[%s]" % (self.parent, self.predicate)

    @property
    def schema(self):
        return self.parent.schema


class ColumnWise(TableExpr):
    """

    a op b
    """
    __hash__ = Expr.__hash__

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __eq__(self, other):
        return Eq(self, other)

    def __lt__(self, other):
        return LT(self, other)

    def __gt__(self, other):
        return GT(self, other)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    def __div__(self, other):
        return Div(self, other)

    def __rdiv__(self, other):
        return Div(other, self)

    def __sub_(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __pow__(self, other):
        return Pow(self, other)

    def __rpow__(self, other):
        return Pow(other, self)

    def __mod__(self, other):
        return Mod(self, other)

    def __rmod__(self, other):
        return Mod(other, self)

    def count(self):
        return count(self)

    def nunique(self):
        return nunique(self)

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


class BinOp(ColumnWise):
    """ A column-wise Binary Operation

    >>> t = TableSymbol('{name: string, amount: int, id: int}')

    >>> data = [['Alice', 100, 1],
    ...         ['Bob', 200, 2],
    ...         ['Alice', 50, 3]]

    >>> from blaze.compute.python import compute
    >>> list(compute(t['amount'] * 10, data))
    [1000, 2000, 500]
    """
    __slots__ = 'lhs', 'rhs'

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        left = _str(self.lhs)
        right = _str(self.rhs)
        if builtins.any(s in left for s in operators):
            left = '(%s)' % left
        if builtins.any(s in right for s in operators):
            right = '(%s)' % right
        return '%s %s %s' % (left, self.symbol, right)


class Boolean(BinOp):  # TODO: support Not
    @property
    def schema(self):
        return dshape('bool')


class And(Boolean):
    symbol = '&'
    op = operator.and_


class Or(Boolean):
    symbol = '|'
    op = operator.or_


class Relational(Boolean):
    pass


class Eq(Relational):
    symbol = '=='
    op = operator.eq


class GT(Relational):
    symbol = '>'
    op = operator.gt


class LT(Relational):
    symbol = '<'
    op = operator.lt


class Arithmetic(BinOp):
    @property
    def schema(self):
        # TODO: Infer schema based on input types
        return dshape('real')


class Add(Arithmetic):
    symbol = '+'
    op = operator.add


class Mul(Arithmetic):
    symbol = '*'
    op = operator.mul


class Sub(Arithmetic):
    symbol = '-'
    op = operator.sub


class Div(Arithmetic):
    symbol = '/'
    op = operator.truediv


class Pow(Arithmetic):
    symbol = '**'
    op = operator.pow


class Mod(Arithmetic):
    symbol = '%'
    op = operator.mod


class Join(TableExpr):
    """ Join two tables on common columns

    Parameters
    ----------
    lhs : TableExpr
    rhs : TableExpr
    on_left : string
    on_right : string

    >>> names = TableSymbol('{name: string, id: int}')
    >>> amounts = TableSymbol('{amount: int, id: int}')

    Join tables based on shared column name
    >>> joined = Join(names, amounts, 'id')

    Join based on different column names
    >>> amounts = TableSymbol('{amount: int, acctNumber: int}')
    >>> joined = Join(names, amounts, 'id', 'acctNumber')
    """
    __slots__ = 'lhs', 'rhs', 'on_left', 'on_right'

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


class UnaryOp(ColumnWise):
    """ A column-wise Unary Operation

    >>> t = TableSymbol('{name: string, amount: int, id: int}')

    >>> data = [['Alice', 100, 1],
    ...         ['Bob', 200, 2],
    ...         ['Alice', 50, 3]]

    >>> from blaze.compute.python import compute
    >>> list(compute(log(t['amount']), data))  # doctest: +SKIP
    [4.605170185988092, 5.298317366548036, 3.912023005428146]
    """
    __slots__ = 'parent',

    def __init__(self, table):
        self.parent = table

    def __str__(self):
        return '%s(%s)' % (self.symbol, self.parent)

    @property
    def symbol(self):
        return type(self).__name__

class sin(UnaryOp): pass
class cos(UnaryOp): pass
class tan(UnaryOp): pass
class exp(UnaryOp): pass
class log(UnaryOp): pass


class Reduction(Scalar):
    """ A column-wise reduction

    >>> t = TableSymbol('{name: string, amount: int, id: int}')
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

    >>> t = TableSymbol('{name: string, amount: int, id: int}')
    >>> e = By(t, t['name'], t['amount'].sum())

    >>> data = [['Alice', 100, 1],
    ...         ['Bob', 200, 2],
    ...         ['Alice', 50, 3]]

    >>> from blaze.compute.python import compute
    >>> compute(e, data) #doctest: +SKIP
    {'Alice': 150, 'Bob': 200}
    """

    __slots__ = 'parent', 'grouper', 'apply'

    def __init__(self, parent, grouper, apply):
        self.parent = parent
        s = TableSymbol(parent.schema)
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
    __slots__ = 'parent', 'column', 'ascending'

    def __init__(self, parent, column, ascending=True):
        self.parent = parent
        self.column = column
        self.ascending = ascending

    @property
    def schema(self):
        return self.parent.schema


class Head(TableExpr):
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
