from __future__ import absolute_import, division, print_function

import datashape
from datashape import (discover, Tuple, Record, dshape, Fixed, DataShape,
    to_numpy_dtype, isdimension, var)
from datashape.predicates import iscollection, isscalar, isrecord
from pandas import DataFrame, Series
import itertools
from functools import reduce
import numpy as np
import warnings
from collections import Iterator

from .expr import Expr, Symbol, ndim
from .dispatch import dispatch
from into import into, resource
from .compatibility import _strtypes, unicode

__all__ = ['Data', 'Table', 'into', 'to_html']

names = ('_%d' % i for i in itertools.count(1))


not_an_iterator = []

try:
    import bcolz
    not_an_iterator.append(bcolz.carray)
except ImportError:
    pass
try:
    import pymongo
    not_an_iterator.append(pymongo.collection.Collection)
except ImportError:
    pass


def Data(data, dshape=None, name=None, fields=None, columns=None,
         schema=None, **kwargs):
    sub_uri = ''
    if isinstance(data, _strtypes):
        if '::' in data:
            data, sub_uri = data.split('::')
        data = resource(data, schema=schema, dshape=dshape,
                              columns=columns, **kwargs)
    if (isinstance(data, Iterator) and
            not isinstance(data, tuple(not_an_iterator))):
        data = tuple(data)
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
            dshape = DataShape(*(dshape.shape[:-1] + (schema,)))
        elif isrecord(dshape.measure) and fields:
            types = dshape.measure.types
            schema = Record(list(zip(fields, types)))
            dshape = DataShape(*(dshape.shape + (schema,)))

    ds = datashape.dshape(dshape)

    if (hasattr(data, 'schema')
         and isinstance(data.schema, (DataShape, str, unicode))
         and ds.measure != data.dshape.measure):
        raise TypeError('%s schema %s does not match schema %s' %
                        (type(data).__name__, data.schema,
                                              ds.measure))

    name = name or next(names)
    result = InteractiveSymbol(data, ds, name)

    if sub_uri:
        for field in sub_uri.split('/'):
            if field:
                result = result[field]

    return result


class InteractiveSymbol(Symbol):
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

    def __init__(self, data, dshape, name=None):
        self.data = data
        self.dshape = dshape
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


@dispatch(InteractiveSymbol, dict)
def _subs(o, d):
    return o


@dispatch(Expr)
def compute(expr, **kwargs):
    resources = expr._resources()
    if not resources:
        raise ValueError("No data resources found")
    else:
        return compute(expr, resources, **kwargs)


def concrete_head(expr, n=10):
    """ Return head of computed expression """
    if not expr._resources():
        raise ValueError("Expression does not contain data resources")
    if not iscollection(expr.dshape):
        return compute(expr)

    head = expr.head(n + 1)
    result = compute(head)

    if len(result) == 0:
        return DataFrame(columns=expr.fields)
    if isrecord(expr.dshape.measure):
        return into(DataFrame, result, dshape=expr.dshape)
    else:
        df = into(DataFrame, result, dshape=expr.dshape)
        df.columns = [expr._name]
        return df


def repr_tables(expr, n=10):
    result = concrete_head(expr, n)

    if isinstance(result, (DataFrame, Series)):
        s = repr(result)
        if len(result) > 10:
            result = result[:10]
            s = '\n'.join(s.split('\n')[:-1]) + '\n...'
        return s
    else:
        return repr(result) # pragma: no cover


def numel(shape):
    if var in shape:
        return None
    if not shape:
        return 1
    return reduce(lambda x, y: x * y, shape, 1)

def short_dshape(ds, nlines=5):
    s = datashape.coretypes.pprint(ds)
    lines = s.split('\n')
    if len(lines) > 5:
        s = '\n'.join(lines[:nlines]) + '\n  ...'
    return s

def expr_repr(expr, n=10):
    # Pure Expressions, not interactive
    if not expr._resources():
        return str(expr)

    # Scalars
    if ndim(expr) == 0 and isscalar(expr.dshape):
        return repr(compute(expr))

    # Tables
    if ndim(expr) == 1:
        return repr_tables(expr, 10)

    # Smallish arrays
    if ndim(expr) >= 2 and  numel(expr.shape) and numel(expr.shape) < 1000000:
        return repr(compute(expr))

    # Other
    dat = expr._resources().values()
    if len(dat) == 1:
        dat = dat[0]

    s = 'Data:       %s' % dat
    if not isinstance(expr, Symbol):
        s += '\nExpr:       %s' % str(expr)
    s += '\nDataShape:  %s' % short_dshape(expr.dshape, nlines=7)

    return s


@dispatch(DataFrame)
def to_html(df):
    return df.to_html()


@dispatch(Expr)
def to_html(expr):
    # Tables
    if ndim(expr) == 1:
        return to_html(concrete_head(expr))

    return to_html(repr(expr))


@dispatch(object)
def to_html(o):
    return repr(o)


@dispatch(_strtypes)
def to_html(o):
    return o.replace('\n', '<br>')


@dispatch((object, type, str), Expr)
def into(a, b, **kwargs):
    result = compute(b, **kwargs)
    kwargs['dshape'] = b.dshape
    return into(a, result, **kwargs)


def table_length(expr):
    try:
        return expr._len()
    except TypeError:
        return compute(expr.count())


Expr.__repr__ = expr_repr
Expr._repr_html_ = lambda self: to_html(self)
Expr.__len__ = table_length
