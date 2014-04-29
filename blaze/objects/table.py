""" An abstract Table

>>> accounts = Table('{name: string, amount: int}')
>>> deadbeats = accounts['name'][accounts['amount'] < 0]
"""

from __future__ import absolute_import, division, print_function

from datashape import dshape, var, DataShape, Record
import operator


class Table(object):
    __slots__ = 'schema',

    def __init__(self, schema):
        self.schema = dshape(schema)

    @property
    def args(self):
        return tuple(getattr(self, slot) for slot in self.__slots__)

    def __eq__(self, other):
        return type(self) == type(other) and self.args == other.args

    def __hash__(self):
        return hash((type(self), self.args))

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


    def __str__(self):
        return "%s(%s)" % (type(self).__name__, ', '.join(map(str, self.args)))

    __repr__ = __str__


class Projection(Table):
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


class Column(Projection):
    """

    SELECT a
    FROM table
    """
    def __init__(self, table, column):
        self.table = table
        self._columns = (column,)

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


class Selection(Table):
    """
    WHERE a op b
    """
    __slots__ = 'table', 'predicate'

    def __init__(self, table, predicate):
        self.table = table
        self.predicate = predicate  # A Relational

    @property
    def schema(self):
        return self.table.schema


class ColumnWise(Column):
    """

    a op b
    """
    __slots__ = 'lhs', 'rhs'

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

class Relational(ColumnWise):
    @property
    def schema(self):
        return dshape('bool')


class Eq(Relational):
    op = operator.eq


class GT(Relational):
    op = operator.gt


class LT(Relational):
    op = operator.lt

class Arithmetic(ColumnWise):
    @property
    def schema(self):
        # TODO: Infer schema based on input types
        return dshape('real')

class Add(Arithmetic):
    op = operator.add

class Mul(Arithmetic):
    op = operator.mul

class Sub(Arithmetic):
    op = operator.sub

class Div(Arithmetic):
    op = operator.div

class Pow(Arithmetic):
    op = operator.pow

class Mod(Arithmetic):
    op = operator.mod
