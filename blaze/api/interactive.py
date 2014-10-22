from __future__ import absolute_import, division, print_function

import datashape
from datashape import (discover, Tuple, Record, dshape, Fixed, DataShape,
    to_numpy_dtype, isdimension, var)
from datashape.predicates import iscollection, isscalar, isrecord
from pandas import DataFrame, Series
import itertools
import numpy as np
from dynd import nd
import warnings

from ..expr import Expr, Symbol
from ..dispatch import dispatch
from .into import into
from ..compatibility import _strtypes, unicode
from ..resource import resource

__all__ = ['Data', 'Table', 'into', 'to_html']

names = ('_%d' % i for i in itertools.count(1))

class Data(Symbol):
    """ Interactive data

    The ``Data`` object presents a familiar view onto a variety of forms of
    data.  This user-level object provides an interactive experience to using
    Blaze's abstract expressions.

    Parameters
    ----------

    data: anything
        Any type with ``discover`` and ``compute`` implementations
    fields: list of strings - optional
        Field or column names, will be inferred from datasource if possible
    dshape: string or DataShape - optional
        Datashape describing input data
    name: string - optional
        A name for the table

    Examples
    --------

    >>> t = Data([(1, 'Alice', 100),
    ...           (2, 'Bob', -200),
    ...           (3, 'Charlie', 300),
    ...           (4, 'Denis', 400),
    ...           (5, 'Edith', -500)],
    ...          fields=['id', 'name', 'balance'])

    >>> t[t.balance < 0].name
        name
    0    Bob
    1  Edith
    """
    __slots__ = 'data', 'dshape', '_name'

    def __init__(self, data, dshape=None, name=None, fields=None, columns=None,
            schema=None, **kwargs):
        if isinstance(data, _strtypes):
            data = resource(data, schema=schema, dshape=dshape,
                    columns=columns, **kwargs)
        if columns:
            warnings.warn("columns kwarg deprecated.  Use fields instead",
                          DeprecationWarning)
        if columns and not fields:
            fields = columns
        if schema and dshape:
            raise ValueError("Please specify one of schema= or dshape= keyword"
                    " arguments")
        if schema and not dshape:
            dshape = var * schema
        if dshape and isinstance(dshape, _strtypes):
            dshape = datashape.dshape(dshape)
        if not dshape:
            dshape = discover(data)
            types = None
            if isinstance(dshape.measure, Tuple) and fields:
                types = dshape[1].dshapes
                schema = Record(list(zip(fields, types)))
                dshape = DataShape(*(dshape.shape + (schema,)))
            elif isscalar(dshape.measure) and fields:
                types = (dshape.measure,) * int(dshape[-2])
                schema = Record(list(zip(fields, types)))
                dshape = DataShape(*(dshape.shape + (schema,)))
            elif isrecord(dshape.measure) and fields:
                types = dshape.measure.types
                schema = Record(list(zip(fields, types)))
                dshape = DataShape(*(dshape.shape + (schema,)))

        self.dshape = datashape.dshape(dshape)

        self.data = data

        if (hasattr(data, 'schema')
             and isinstance(data.schema, (DataShape, str, unicode))
             and self.schema != data.schema):
            raise TypeError('%s schema %s does not match %s schema %s' %
                            (type(data).__name__, data.schema,
                             type(self).__name__, self.schema))

        self._name = name or next(names)

    def _resources(self):
        return {self: self.data}

    @property
    def _args(self):
        return (id(self.data), self.dshape, self._name)

    def __setstate__(self, state):
        for slot, arg in zip(self.__slots__, state):
            setattr(self, slot, arg)

def Table(*args, **kwargs):
    """ Deprecated, see Data instead """
    warnings.warn("Table is deprecated, use Data instead",
                  DeprecationWarning)
    return Data(*args, **kwargs)


@dispatch(Data, dict)
def _subs(o, d):
    return o


@dispatch(Expr)
def compute(expr):
    resources = expr._resources()
    if not resources:
        raise ValueError("No data resources found")
    else:
        return compute(expr, resources)


def concrete_head(expr, n=10):
    """ Return head of computed expression """
    if not expr._resources():
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
    if not expr._resources():
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


@dispatch(DataFrame)
def to_html(df):
    return df.to_html()

@dispatch(Expr)
def to_html(expr):
    return to_html(concrete_head(expr))


@dispatch(object)
def to_html(o):
    return repr(o)


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
Expr._repr_html_ = lambda self: to_html(self)
Expr.__len__ = table_length
