from __future__ import absolute_import, division, print_function

import datashape
from datashape import (discover, Tuple, Record, dshape, Fixed, DataShape,
    to_numpy_dtype, isdimension, var)
from pandas import DataFrame, Series
import itertools
import numpy as np
from dynd import nd

from ..expr.core import Expr
from ..expr.table import TableSymbol, TableExpr
from ..dispatch import dispatch
from .into import into
from ..compatibility import _strtypes, unicode
from .resource import resource

__all__ = ['Table', 'compute', 'into']

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
    __slots__ = 'data', 'dshape', '_name', 'iscolumn'

    def __init__(self, data, dshape=None, name=None, columns=None,
            iscolumn=False, schema=None):
        if isinstance(data, str):
            data = resource(data)
        if schema and dshape:
            raise ValueError("Please specify one of schema= or dshape= keyword"
                    " arguments")
        if schema and not dshape:
            dshape = var * schema
        if dshape and isinstance(dshape, _strtypes):
            dshape = datashape.dshape(dshape)
        if dshape and not isdimension(dshape[0]):
            dshape = var * dshape
        if not dshape:
            dshape = discover(data)
            types = None
            if isinstance(dshape[1], Tuple):
                columns = columns or list(range(len(dshape[1].dshapes)))
                types = dshape[1].dshapes
            if isinstance(dshape[1], Record):
                columns = columns or dshape[1].names
                types = dshape[1].types
            if isinstance(dshape[1], Fixed):
                types = (dshape[2],) * int(dshape[1])
            if not columns:
                raise TypeError("Could not infer column names from data. "
                                "Please specify column names with `columns=` "
                                "keyword")
            if not types:
                raise TypeError("Could not infer data types from data. "
                                "Please specify schema with `schema=` keyword")

            dshape = dshape[0] * datashape.dshape(Record(list(zip(columns, types))))

        self.dshape = datashape.dshape(dshape)

        self.data = data

        if (hasattr(data, 'schema')
             and isinstance(data.schema, (DataShape, str, unicode))
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
        return (id(self.data), self.dshape, self._name, self.iscolumn)

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

        if not len(result):
            return DataFrame(columns=expr.columns)

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
    f = into.dispatch(a, type(b))
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
    schema = dshape(str(b.schema).replace('?', ''))
    if b.iscolumn:
        return into(np.ndarray(0), compute(b),
                dtype=to_numpy_dtype(schema[0].types[0]))
    else:
        return into(np.ndarray(0), compute(b), dtype=to_numpy_dtype(schema))


def table_length(expr):
    try:
        return expr._len()
    except TypeError:
        return compute(expr.count())


Expr.__repr__ = expr_repr
TableExpr.__repr__ = table_repr
TableExpr.to_html = table_html
TableExpr._repr_html_ = table_html
TableExpr.__len__ = table_length
