from datashape import dshape, var, DataShape, Record


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
        if isinstance(key, (tuple, list)):
            key = tuple(key)
            if not all(col in self.columns for col in key):
                raise ValueError("Mismatched Columns: %s" % str(key))
            return Projection(self, tuple(key))
        else:
            if key not in self.columns:
                raise ValueError("Mismatched Column: %s" % str(key))
            return Column(self, key)


class Projection(Table):
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
    def __init__(self, table, column):
        self.table = table
        self._columns = (column,)

    def __eq__(self, other):
        return Eq(self, other)

    def __lt__(self, other):
        return LT(self, other)

    def __gt__(self, other):
        return GT(self, other)


class Relational(Table):
    __slots__ = 'lhs', 'rhs'

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    @property
    def schema(self):
        return dshape('bool')


class Eq(Relational):
    pass

class GT(Relational):
    pass

class LT(Relational):
    pass

class Selection(Table):
    __slots__ = 'table', 'predicate'
    def __init__(self, table, predicate):
        self.table = table
        self.predicate = predicate

    @property
    def schema(self):
        return self.table.schema

