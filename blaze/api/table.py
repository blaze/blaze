
from datashape import discover, Tuple, Record, dshape
import itertools

from ..expr.core import Expr
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
    __slots__ = 'data', 'schema', 'name', 'iscolumn'

    def __init__(self, data, name=None, columns=None, schema=None,
            iscolumn=False):
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
        self.iscolumn = iscolumn

    def resources(self):
        return {self: self.data}


@dispatch(Table)
def compute(t):
    return t.data


@dispatch(Expr)
def compute(expr):
    resources = expr.resources()
    if not resources:
        raise ValueError("No data resources found")
    else:
        return compute(expr, resources)


def table_repr(expr, n=10):
    if not expr.resources():
        return str(expr)
    from blaze.data.pandas import into, DataFrame
    from blaze.api.into import into
    if isinstance(expr, TableExpr):
        head = expr.head(n)
    result = compute(head)

    if expr.columns:
        return repr(into(DataFrame(columns=expr.columns), result)) + '\n...'
    else:
        return repr(into(DataFrame, result)) + '\n...'


@dispatch(object, TableExpr)
def into(a, b):
    return into(a, compute(b))


TableExpr.__repr__ = table_repr
