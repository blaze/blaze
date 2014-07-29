
from datashape import discover, Tuple, Record, dshape, Fixed
import itertools

from ..expr.core import Expr
from ..expr.table import TableSymbol, TableExpr
from ..dispatch import dispatch
from ..data.pandas import into, DataFrame
from .into import into

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
    __slots__ = 'data', 'schema', '_name', 'iscolumn'

    def __init__(self, data, name=None, columns=None, schema=None,
            iscolumn=False):
        if not schema:
            schema = discover(data).subshape[0]
            types = None
            if isinstance(schema[0], Tuple):
                columns = columns or list(range(len(schema[0].dshapes)))
                types = schema[0].dshapes
            if isinstance(schema[0], Record):
                columns = columns or schema[0].names
                types = schema[0].types
            if isinstance(schema[0], Fixed):
                types = (schema[1],) * int(schema[0])
            if not columns:
                raise TypeError("Could not infer column names from data. "
                                "Please specify column names with `column=` "
                                "keyword")
            if not types:
                raise TypeError("Could not infer data types from data. "
                                "Please specify schema with `schema=` keyword")

            schema = dshape(Record(list(zip(columns, types))))
        self.schema = dshape(schema)

        self.data = data

        if hasattr(data, 'schema') and self.schema != data.schema:
            raise TypeError('%s schema %s does not match %s schema %s' %
                            (type(data).__name__, data.schema,
                             type(self).__name__, self.schema))

        self._name = name or next(names)
        self.iscolumn = iscolumn

    def resources(self):
        return {self: self.data}

    @property
    def args(self):
        return (id(self.data), self.schema, self._name, self.iscolumn)


@dispatch(Table, dict)
def _subs(o, d):
    return o


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
    if isinstance(expr, TableExpr):
        head = expr.head(n + 1)
        result = compute(head)

        if expr.columns:
            df = into(DataFrame(columns=expr.columns), result)
        else:
            df = into(DataFrame, result)
        s = repr(df)
        if len(df) > 10:
            df = df[:10]
            s = '\n'.join(s.split('\n')[:-1]) + '\n...'
        return s

    else:
        return repr(compute(expr))


@dispatch((type, object), TableExpr)
def into(a, b):
    return into(a, compute(b))


@dispatch(DataFrame, TableExpr)
def into(a, b):
    return into(DataFrame(columns=b.columns), compute(b))

Expr.__repr__ = table_repr
