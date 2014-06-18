
from datashape import discover, Tuple, Record, dshape
import itertools

from ..expr.table import TableSymbol, TableExpr
from ..data.python import Python
from ..dispatch import dispatch
from ..data.core import DataDescriptor, discover

names = ('_%d' % i for i in itertools.count(1))

class Table(TableSymbol):
    """ Interactive Table

    Parameters
    ----------

    data: DataDescriptor, tuple, DataFrame, RDD, SQL Table, ...
        Anything that ``compute`` knows how to work with

    Optional
    --------

    name: string
        A name for the table
    columns: iterable of strings
        Column names, will be inferred from datasource if possible
    schema: string or DataShape
        Explitit Record containing datatypes and column names
    """
    __slots__ = 'data', 'schema', 'name'

    def __init__(self, data, name=None, columns=None, schema=None):
        schema = schema or discover(data).subshape[0]
        if isinstance(schema[0], Tuple):
            columns = columns or list(range(len(schema[0].dshapes)))
            types = schema[0].dshapes
        if isinstance(schema[0], Record):
            columns = columns or schema[0].names
            types = schema[0].types
        self.schema = dshape(Record(list(zip(columns, types))))

        self.data = data
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
