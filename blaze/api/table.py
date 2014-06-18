from ..expr.table import TableSymbol, TableExpr
from ..data.core import DataDescriptor
from ..data.python import Python
from ..dispatch import dispatch
from datashape import discover, Tuple, Record, dshape
import itertools

names = ('_%d' % i for i in itertools.count(1))

class Table(TableSymbol):
    """ Interactive Table """
    __slots__ = 'data', 'schema', 'name'

    @dispatch(DataDescriptor)
    def __init__(self, dd, name=None):
        self.data = dd
        self.schema = dd.schema
        self.name = name or next(names)

    @dispatch((list, tuple))
    def __init__(self, seq, name=None, columns=None):
        schema = discover(seq[:10]).subshape[0]
        if isinstance(schema[0], Tuple):
            columns = columns or list(range(len(schema[0].dshapes)))
            types = schema[0].dshapes
        if isinstance(schema[0], Record):
            columns = columns or schema[0].names
            types = schema[0].types
        self.schema = dshape(Record(list(zip(columns, types))))
        self.data = tuple(seq)
        self.name = name or next(names)

    def resources(self):
        return {self: self.data}


@dispatch(Table)
def compute(t):
    return t.data


@dispatch(TableExpr)
def compute(expr):
    resources = expr.resources()
    if not resources:
        raise ValueError("No data resources found")

    else:
        print(resources)
        return compute(expr, resources)
