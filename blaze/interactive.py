from __future__ import absolute_import, division, print_function

import datashape
import datetime
import operator
import itertools
import warnings

from collections import Iterator
from functools import reduce

from datashape import discover, Tuple, Record, DataShape, var
from datashape.predicates import iscollection, isscalar, isrecord, istabular

from pandas import DataFrame, Series

import numpy as np

from odo import resource, odo
from odo.utils import ignoring

from .expr import Expr, Symbol, ndim
from .dispatch import dispatch
from .compatibility import _strtypes


__all__ = ['Data', 'Table', 'into', 'to_html']


names = ('_%d' % i for i in itertools.count(1))
not_an_iterator = []


with ignoring(ImportError):
    import bcolz
    not_an_iterator.append(bcolz.carray)


with ignoring(ImportError):
    import pymongo
    not_an_iterator.append(pymongo.collection.Collection)
    not_an_iterator.append(pymongo.database.Database)


def Data(data, dshape=None, name=None, fields=None, columns=None, schema=None,
         **kwargs):
    if columns:
        raise ValueError("columns argument deprecated, use fields instead")
    if schema and dshape:
        raise ValueError("Please specify one of schema= or dshape= keyword"
                         " arguments")

    sub_uri = ''
    if isinstance(data, _strtypes):
        if '::' in data:
            data, sub_uri = data.split('::')
        data = resource(data, schema=schema, dshape=dshape, columns=columns,
                        **kwargs)
    if (isinstance(data, Iterator) and
            not isinstance(data, tuple(not_an_iterator))):
        data = tuple(data)
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
            ds = discover(data)
            assert isrecord(ds.measure)
            names = ds.measure.names
            if names != fields:
                raise ValueError('data column names %s\n'
                                 '\tnot equal to fields parameter %s,\n'
                                 '\tuse Data(data).relabel(%s) to rename '
                                 'fields' % (names,
                                             fields,
                                             ', '.join('%s=%r' % (k, v)
                                                       for k, v in
                                                       zip(names, fields))))
            types = dshape.measure.types
            schema = Record(list(zip(fields, types)))
            dshape = DataShape(*(dshape.shape + (schema,)))

    ds = datashape.dshape(dshape)
    result = InteractiveSymbol(data, ds, name)

    if sub_uri:
        for field in sub_uri.split('/'):
            if field:
                result = result[field]

    return result


class InteractiveSymbol(Symbol):
    """Interactive data.

    The ``Data`` object presents a familiar view onto a variety of forms of
    data.  This user-level object provides an interactive experience to using
    Blaze's abstract expressions.

    Parameters
    ----------
    data : object
        Any type with ``discover`` and ``compute`` implementations
    fields : list, optional
        Field or column names, will be inferred from datasource if possible
    dshape : str or DataShape, optional
        DataShape describing input data
    name : str, optional
        A name for the data.

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
        self._name = name or (next(names)
                              if isrecord(dshape.measure)
                              else None)

    def _resources(self):
        return {self: self.data}

    @property
    def _args(self):
        return id(self.data), self.dshape, self._name

    def __setstate__(self, state):
        for slot, arg in zip(self.__slots__, state):
            setattr(self, slot, arg)


Data.__doc__ = InteractiveSymbol.__doc__


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

    if not iscollection(expr.dshape):
        return odo(head, object)
    elif isrecord(expr.dshape.measure):
        return odo(head, DataFrame)
    else:
        df = odo(head, DataFrame)
        df.columns = [expr._name]
        return df
    result = compute(head)

    if len(result) == 0:
        return DataFrame(columns=expr.fields)
    if isrecord(expr.dshape.measure):
        return odo(result, DataFrame, dshape=expr.dshape)
    else:
        df = odo(result, DataFrame, dshape=expr.dshape)
        df.columns = [expr._name]
        return df


def repr_tables(expr, n=10):
    result = concrete_head(expr, n).rename(columns={None: ''})

    if isinstance(result, (DataFrame, Series)):
        s = repr(result)
        if len(result) > 10:
            s = '\n'.join(s.split('\n')[:-1]) + '\n...'
        return s
    else:
        return repr(result)  # pragma: no cover


def numel(shape):
    if var in shape:
        return None
    if not shape:
        return 1
    return reduce(operator.mul, shape, 1)


def short_dshape(ds, nlines=5):
    s = datashape.coretypes.pprint(ds)
    lines = s.split('\n')
    if len(lines) > 5:
        s = '\n'.join(lines[:nlines]) + '\n  ...'
    return s


def coerce_to(typ, x):
    try:
        return typ(x)
    except TypeError:
        return odo(x, typ)


def coerce_scalar(result, dshape):
    if 'float' in dshape:
        return coerce_to(float, result)
    elif 'int' in dshape:
        return coerce_to(int, result)
    elif 'bool' in dshape:
        return coerce_to(bool, result)
    elif 'datetime' in dshape:
        return coerce_to(datetime.datetime, result)
    elif 'date' in dshape:
        return coerce_to(datetime.date, result)
    else:
        return result


def expr_repr(expr, n=10):
    # Pure Expressions, not interactive
    if not expr._resources():
        return str(expr)

    # Scalars
    if ndim(expr) == 0 and isscalar(expr.dshape):
        return repr(coerce_scalar(compute(expr), str(expr.dshape)))

    # Tables
    if (ndim(expr) == 1 and (istabular(expr.dshape) or
                             isscalar(expr.dshape.measure))):
        return repr_tables(expr, 10)

    # Smallish arrays
    if ndim(expr) >= 2 and numel(expr.shape) and numel(expr.shape) < 1000000:
        return repr(compute(expr))

    # Other
    dat = expr._resources().values()
    if len(dat) == 1:
        dat = list(dat)[0]  # may be dict_values

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
    if not expr._resources() or ndim(expr) != 1:
        return to_html(repr(expr))
    return to_html(concrete_head(expr))


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
    except ValueError:
        return compute(expr.count())


Expr.__repr__ = expr_repr
Expr._repr_html_ = lambda x: to_html(x)
Expr.__len__ = table_length


def intonumpy(data, dtype=None, **kwargs):
    # TODO: Don't ignore other kwargs like copy
    result = odo(data, np.ndarray)
    if dtype and result.dtype != dtype:
        result = result.astype(dtype)
    return result


def convert_base(typ, x):
    x = compute(x)
    try:
        return typ(x)
    except:
        return typ(odo(x, typ))

Expr.__array__ = intonumpy
Expr.__int__ = lambda x: convert_base(int, x)
Expr.__float__ = lambda x: convert_base(float, x)
Expr.__complex__ = lambda x: convert_base(complex, x)
Expr.__bool__ = lambda x: convert_base(bool, x)
Expr.__nonzero__ = lambda x: convert_base(bool, x)
Expr.__iter__ = into(Iterator)
