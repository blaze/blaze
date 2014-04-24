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


class Projection(Table):
    __slots__ = 'table', '_columns'

    def __init__(self, table, columns):
        self.table = table
        self._columns = columns

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
