""" An abstract Table

>>> accounts = TableSymbol('{name: string, amount: int}')
>>> deadbeats = accounts['name'][accounts['amount'] < 0]
"""

from __future__ import absolute_import, division, print_function

from datashape import dshape, var, DataShape, Record
import operator
from .core import Expr

class TableExpr(Expr):
    @property
    def dshape(self):
        return var * self.schema

    @property
    def columns(self):
        return self.schema[0].names

    def __getitem__(self, key):
        if isinstance(key, Relational):
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


class TableSymbol(TableExpr):
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
    __slots__ = 'table', '_columns'

    def __init__(self, table, columns):
        self.table = table
        self._columns = tuple(columns)

    @property
    def columns(self):
        return self._columns

    @property
    def schema(self):
        d = self.table.schema[0].fields
        return DataShape(Record([(col, d[col]) for col in self.columns]))

    def __str__(self):
        return '%s[%s]' % (self.table,
                           ', '.join(["'%s'" % col for col in self.columns]))


class Column(Projection):
    """

    SELECT a
    FROM table
    """
    def __init__(self, table, column):
        self.table = table
        self._columns = (column,)

    def __str__(self):
        return "%s['%s']" % (self.table, self.columns[0])

    @property
    def schema(self):
        return dshape(self.table.schema[0][self.columns[0]])

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


class Selection(TableExpr):
    """
    WHERE a op b
    """
    __slots__ = 'table', 'predicate'

    def __init__(self, table, predicate):
        self.table = table
        self.predicate = predicate  # A Relational

    def __str__(self):
        return "%s[%s]" % (self.table, self.predicate)

    @property
    def schema(self):
        return self.table.schema


class ColumnWise(Column):
    """

    a op b
    """
    pass


class BinOp(ColumnWise):
    __slots__ = 'lhs', 'rhs'
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return '%s %s %s' % (self.lhs, self.symbol, self.rhs)


class Relational(BinOp):
    @property
    def schema(self):
        return dshape('bool')


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
    __slots__ = 'table',

    def __init__(self, table):
        self.table = table

    def __str__(self):
        return '%s(%s)' % (self.symbol, self.table)

    @property
    def symbol(self):
        return type(self).__name__

class sin(UnaryOp): pass
class cos(UnaryOp): pass
class tan(UnaryOp): pass
class exp(UnaryOp): pass
class log(UnaryOp): pass


class Reduction(ColumnWise):
    __slots__ = 'table', 'reduction'

    def __init__(self, table, reduction):
        self.table = table
        self.reduction = reduction

    @property
    def dshape(self):
        return dshape(self.table.dshape.subarray(1))
