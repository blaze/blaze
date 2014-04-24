from datashape import dshape, var, DataShape, Record


class Table(object):
    def __init__(self, schema):
        self.schema = dshape(schema)

    @property
    def dshape(self):
        return var * self.schema

    @property
    def columns(self):
        return self.schema[0].names

class Projection(Table):
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
