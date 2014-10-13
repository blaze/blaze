from __future__ import absolute_import, division, print_function

import datashape
from datashape import (discover, Tuple, Record, dshape, Fixed, DataShape,
    to_numpy_dtype, isdimension, var)
from datashape.predicates import iscollection, isscalar
from pandas import DataFrame, Series
import itertools
import numpy as np
from dynd import nd

from ..expr import TableSymbol, Expr, Symbol
from ..dispatch import dispatch
from .into import into
from ..compatibility import _strtypes, unicode
from ..resource import resource

__all__ = ['Table', 'compute', 'into']

names = ('_%d' % i for i in itertools.count(1))

class Table(Symbol):
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
    __slots__ = 'data', 'dshape', '_name'

    def __init__(self, data, dshape=None, name=None, columns=None,
            schema=None):
        if isinstance(data, str):
            data = resource(data, schema=schema, dshape=dshape, columns=columns)
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
            if not types:
                types = datashape.dshape(dshape[-1])
            if not columns:
                raise TypeError("Could not infer column names from data. "
                                "Please specify column names with `columns=` "
                                "keyword")

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

    def resources(self):
        return {self: self.data}

    @property
    def _args(self):
        return (id(self.data), self.dshape, self._name)

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
    if iscollection(expr.dshape):
        head = expr.head(n + 1)
        result = compute(head)

        if not len(result):
            return DataFrame(columns=expr.fields)

        if iscollection(expr.dshape):
            return into(DataFrame(columns=expr.fields), result)
    else:
        return compute(expr)


def expr_repr(expr, n=10):
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
        return repr(result) # pragma: no cover


def expr_html(expr, n=10):
    return concrete_head(expr).to_html()


@dispatch(type, Expr)
def into(a, b, **kwargs):
    f = into.dispatch(a, type(b))
    return f(a, b, **kwargs)


@dispatch(object, Expr)
def into(a, b, **kwargs):
    return into(a, compute(b), dshape=kwargs.pop('dshape', b.dshape),
                schema=b.schema, **kwargs)


@dispatch(DataFrame, Expr)
def into(a, b, **kwargs):
    return into(DataFrame(columns=b.fields), compute(b))


@dispatch(nd.array, Expr)
def into(a, b, **kwargs):
    return into(nd.array(), compute(b), dtype=str(b.schema))


@dispatch(np.ndarray, Expr)
def into(a, b, **kwargs):
    schema = dshape(str(b.schema).replace('?', ''))
    return into(np.ndarray(0), compute(b), dtype=to_numpy_dtype(schema))


def table_length(expr):
    try:
        return expr._len()
    except TypeError:
        return compute(expr.count())


Expr.__repr__ = expr_repr
Expr.to_html = expr_html
Expr._repr_html_ = expr_html
Expr.__len__ = table_length
