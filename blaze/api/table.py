
from datashape import (discover, Tuple, Record, dshape, Fixed, DataShape,
    to_numpy_dtype)
from pandas import DataFrame, Series
import itertools
import numpy as np
from dynd import nd

from ..expr.core import Expr
from ..expr.table import TableSymbol, TableExpr
from ..dispatch import dispatch
from ..data.pandas import into
from .into import into

names = ('_%d' % i for i in itertools.count(1))

class Table(TableSymbol):
    """ Table of data

    The ``Table`` object presents a familiar view onto a variety of forms of
    data.  This user-level object provides an interactive experience to using
    Blaze's abstract expressions.

    Parameters
    ----------

    data: anything
        Any type with ``discover`` and ``compute`` implementations
    columns: list of strings - optional
        Column names, will be inferred from datasource if possible
    schema: string or DataShape - optional
        Explicit Record containing datatypes and column names
    name: string - optional
        A name for the table

    Examples
    --------

    >>> t = Table([(1, 'Alice', 100),
    ...            (2, 'Bob', -200),
    ...            (3, 'Charlie', 300),
    ...            (4, 'Denis', 400),
    ...            (5, 'Edith', -500)],
    ...            columns=['id', 'name', 'balance'])

    >>> t[t.balance < 0].name
        name
    0    Bob
    1  Edith
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

        if (hasattr(data, 'schema')
             and isinstance(data.schema, (DataShape, str))
             and self.schema != data.schema):
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

    def __setstate__(self, state):
        for slot, arg in zip(self.__slots__, state):
            setattr(self, slot, arg)


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


def concrete_head(expr, n=10):
    """ Return head of computed expression """
    if not expr.resources():
        raise ValueError("Expression does not contain data resources")
    if isinstance(expr, TableExpr):
        head = expr.head(n + 1)
        result = compute(head)

        if expr.columns:
            return into(DataFrame(columns=expr.columns), result)
        else:
            return into(DataFrame, result)
    else:
        return repr(compute(expr))


def table_repr(expr, n=10):
    if not expr.resources():
        return str(expr)

    result = concrete_head(expr, n)

    if isinstance(result, (DataFrame, Series)):
        s = repr(result)
        if len(result) > 10:
            result = result[:10]
            s = '\n'.join(s.split('\n')[:-1]) + '\n...'
        return s
    else:
        return repr(result)


def expr_repr(expr):
    if not expr.resources():
        return str(expr)
    else:
        return str(compute(expr))


def table_html(expr, n=10):
    return concrete_head(expr).to_html()


@dispatch(type, TableExpr)
def into(a, b, **kwargs):
    f = into.resolve((a, type(b)))
    return f(a, b, **kwargs)


@dispatch(object, TableExpr)
def into(a, b):
    return into(a, compute(b))


@dispatch(DataFrame, TableExpr)
def into(a, b):
    return into(DataFrame(columns=b.columns), compute(b))


@dispatch(nd.array, TableExpr)
def into(a, b):
    return into(nd.array(), compute(b), dtype=str(b.schema))


@dispatch(np.ndarray, TableExpr)
def into(a, b):
    if b.iscolumn:
        return into(np.ndarray(0), compute(b),
                dtype=to_numpy_dtype(b.schema[0].types[0]))
    else:
        return into(np.ndarray(0), compute(b), dtype=to_numpy_dtype(b.schema))


Expr.__repr__ = expr_repr
TableExpr.__repr__ = table_repr
TableExpr.to_html = table_html
TableExpr._repr_html_ = table_html
