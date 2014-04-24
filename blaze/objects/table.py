from datashape import dshape, var


class Table(object):
    def __init__(self, schema):
        self.schema = dshape(schema)

    @property
    def dshape(self):
        return var * self.schema

    @property
    def columns(self):
        return self.schema[0].names
