from __future__ import absolute_import, division, print_function

from collections import Iterator, Mapping
import decimal
import datetime
from functools import reduce, partial
import itertools
import operator
import warnings

import datashape
from datashape import discover, Tuple, Record, DataShape, var, Map
from datashape.predicates import isscalar, iscollection, isrecord, istabular, _dimensions
import numpy as np
from odo import resource, odo, append, drop
from odo.utils import ignoring, copydoc
from odo.compatibility import unicode
from pandas import DataFrame, Series, Timestamp


from .expr import Expr, Symbol, ndim
from .expr.expressions import sanitized_dshape
from .dispatch import dispatch
from .compatibility import _strtypes


__all__ = ['Data', 'into', 'to_html', 'data']


names = ('_%d' % i for i in itertools.count(1))
not_an_iterator = []


with ignoring(ImportError):
    import bcolz
    not_an_iterator.append(bcolz.carray)


with ignoring(ImportError):
    import pymongo
    not_an_iterator.append(pymongo.collection.Collection)
    not_an_iterator.append(pymongo.database.Database)


class _Data(Symbol):

    # NOTE: This docstring is meant to correspond to the ``data()`` API, which
    # is why the Parameters section doesn't match the arguments to
    # ``_Data.__init__()``.

    """Bind a data resource to a symbol, for use in expressions and
    computation.

    A ``data`` object presents a consistent view onto a variety of concrete
    data sources.  Like ``symbol`` objects, they are meant to be used in
    expressions.  Because they are tied to concrete data resources, ``data``
    objects can be used with ``compute`` directly, making them convenient for
    interactive exploration.

    Parameters
    ----------
    data_source : object
        Any type with ``discover`` and ``compute`` implementations
    fields : list, optional
        Field or column names, will be inferred from data_source if possible
    dshape : str or DataShape, optional
        DataShape describing input data
    name : str, optional
        A name for the data.

    Examples
    --------
    >>> t = data([(1, 'Alice', 100),
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
    __slots__ = '_hash', 'data', 'dshape', '_name'

    def __init__(self, data_source, dshape, name=None):
        self.data = data_source
        self.dshape = dshape
        self._name = name or (next(names)
                              if isrecord(dshape.measure)
                              else None)
        self._hash = None

    def _resources(self):
        return {self: self.data}

    @property
    def _hashargs(self):
        data = self.data
        try:
            # cannot use isinstance(data, Hashable)
            # some classes give a false positive
            hash(data)
        except TypeError:
            data = id(data)
        return data, self.dshape, self._name

    def __repr__(self):
        fmt =  "<'{}' data; _name='{}', dshape='{}'>"
        return fmt.format(type(self.data).__name__,
                          self._name,
                          sanitized_dshape(self.dshape))


class InteractiveSymbol(_Data):
    """Deprecated and replaced by the _Data class. ``InteractiveSymbol`` will
    be removed in version 0.10.  ``data`` is the public API for creating
    ``Data`` objects.
    """

    def __new__(cls, *args, **kwargs):
        warnings.warn("InteractiveSymbol has been deprecated in 0.10 and will be removed in 0.11.  Use ``data`` to create ``Data`` objects directly.", DeprecationWarning)
        return data(*args, **kwargs)


def Data(data_source, dshape=None, name=None, fields=None, schema=None, **kwargs):
    warnings.warn("""The `Data` callable has been deprecated in 0.10 and will be removed in
                   version >= 0.11. It has been renamed `data`.""", DeprecationWarning)
    return data(data_source, dshape=dshape, name=name, fields=fields, schema=schema, **kwargs)


@copydoc(_Data)
def data(data_source, dshape=None, name=None, fields=None, schema=None, **kwargs):
    if schema and dshape:
        raise ValueError("Please specify one of schema= or dshape= keyword"
                         " arguments")

    if isinstance(data_source, _Data):
        return data(data_source.data, dshape, name, fields, schema, **kwargs)

    if isinstance(data_source, _strtypes):
        data_source = resource(data_source, schema=schema, dshape=dshape, **kwargs)
        return _Data(data_source, discover(data_source), name)

    if (isinstance(data_source, Iterator) and
            not isinstance(data_source, tuple(not_an_iterator))):
        data_source = tuple(data_source)
    if schema and not dshape:
        dshape = var * schema
    if dshape and isinstance(dshape, _strtypes):
        dshape = datashape.dshape(dshape)
    if not dshape:
        dshape = discover(data_source)
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
            ds = discover(data_source)
            assert isrecord(ds.measure)
            names = ds.measure.names
            if names != fields:
                raise ValueError('data column names %s\n'
                                 '\tnot equal to fields parameter %s,\n'
                                 '\tuse data(data_source).relabel(%s) to rename '
                                 'fields' % (names,
                                             fields,
                                             ', '.join('%s=%r' % (k, v)
                                                       for k, v in
                                                       zip(names, fields))))
            types = dshape.measure.types
            schema = Record(list(zip(fields, types)))
            dshape = DataShape(*(dshape.shape + (schema,)))

    ds = datashape.dshape(dshape)
    return _Data(data_source, ds, name)


@dispatch(_Data, Mapping)
def _subs(o, d):
    return o


@dispatch(Expr)
def compute(expr, **kwargs):
    resources = expr._resources()
    if not resources:
        raise ValueError("No data resources found")
    else:
        return compute(expr, resources, **kwargs)


@dispatch(Expr, _Data)
def compute_down(expr, dta, **kwargs):
    return compute(expr, dta.data, **kwargs)


@dispatch(Expr, _Data)
def pre_compute(expr, dta, **kwargs):
    return pre_compute(expr, dta.data, **kwargs)


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


def _peek_tables(expr, n=10):
    return concrete_head(expr, n).rename(columns={None: ''})


def repr_tables(expr, n=10):
    result = concrete_head(expr, n).rename(columns={None: ''})

    if isinstance(result, (DataFrame, Series)):
        s = repr(result)
        if len(result) > 10:
            s = '\n'.join(s.split('\n')[:-1]) + '\n...'
        return s
    else:
        return result.peek()  # pragma: no cover


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


def coerce_to(typ, x, odo_kwargs=None):
    try:
        return typ(x)
    except TypeError:
        return odo(x, typ, **(odo_kwargs or {}))


def coerce_scalar(result, dshape, odo_kwargs=None):
    dshape = str(dshape)
    coerce_ = partial(coerce_to, x=result, odo_kwargs=odo_kwargs)
    if 'float' in dshape:
        return coerce_(float)
    if 'decimal' in dshape:
        return coerce_(decimal.Decimal)
    elif 'int' in dshape:
        return coerce_(int)
    elif 'bool' in dshape:
        return coerce_(bool)
    elif 'datetime' in dshape:
        return coerce_(Timestamp)
    elif 'date' in dshape:
        return coerce_(datetime.date)
    elif 'timedelta' in dshape:
        return coerce_(datetime.timedelta)
    else:
        return result


def coerce_core(result, dshape, odo_kwargs=None):
    """Coerce data to a core data type."""
    if iscoretype(result):
        return result
    elif isscalar(dshape):
        result = coerce_scalar(result, dshape, odo_kwargs=odo_kwargs)
    elif istabular(dshape) and isrecord(dshape.measure):
        result = into(DataFrame, result, **(odo_kwargs or {}))
    elif iscollection(dshape):
        dim = _dimensions(dshape)
        if dim == 1:
            result = into(Series, result, **(odo_kwargs or {}))
        elif dim > 1:
            result = into(np.ndarray, result, **(odo_kwargs or {}))
        else:
            msg = "Expr with dshape dimensions < 1 should have been handled earlier: dim={}"
            raise ValueError(msg.format(str(dim)))
    else:
        msg = "Expr does not evaluate to a core return type"
        raise ValueError(msg)

    return result


def _peek(expr):
    # Pure Expressions, not interactive
    if not set(expr._resources().keys()).issuperset(expr._leaves()):
        return expr

    # Scalars
    if ndim(expr) == 0 and isscalar(expr.dshape):
        return coerce_scalar(compute(expr), str(expr.dshape))

    # Tables
    if (ndim(expr) == 1 and (istabular(expr.dshape) or
                             isscalar(expr.dshape.measure) or
                             isinstance(expr.dshape.measure, Map))):
        return _peek_tables(expr, 10)

    # Smallish arrays
    if ndim(expr) >= 2 and numel(expr.shape) and numel(expr.shape) < 1000000:
        return compute(expr)

    # Other
    dat = expr._resources().values()
    if len(dat) == 1:
        dat = list(dat)[0]  # may be dict_values
    return dat


def expr_repr(expr, n=10):
    # Pure Expressions, not interactive
    if not set(expr._resources().keys()).issuperset(expr._leaves()):
        return str(expr)

    # Scalars
    if ndim(expr) == 0 and isscalar(expr.dshape):
        return repr(coerce_scalar(compute(expr), str(expr.dshape)))

    # Tables
    if (ndim(expr) == 1 and (istabular(expr.dshape) or
                             isscalar(expr.dshape.measure) or
                             isinstance(expr.dshape.measure, Map))):
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
        return to_html(expr_repr(expr))
    return to_html(concrete_head(expr))


@dispatch(object)
def to_html(o):
    return repr(o)


@dispatch(_strtypes)
def to_html(o):
    return o.replace('\n', '<br>')


@dispatch((object, type, str, unicode), Expr)
def into(a, b, **kwargs):
    result = compute(b, return_type='native', **kwargs)
    kwargs['dshape'] = b.dshape
    return into(a, result, **kwargs)


@dispatch((object, type, str, unicode), _Data)
def into(a, b, **kwargs):
    return into(a, b.data, **kwargs)


@dispatch(_Data, object)
def append(a, b, **kwargs):
    return append(a.data, b, **kwargs)


@dispatch(_Data)
def drop(d):
    return drop(d.data)


def table_length(expr):
    try:
        return expr._len()
    except ValueError:
        return int(expr.count())

_warning_msg = """
In version 0.11, Blaze's expresssion repr will return a standard 
representation and will no longer implicitly compute.  Use the `peek()`
method to see a preview of the expression's results.\
"""

use_new_repr = False
def _choose_repr(self):
    if use_new_repr:
        return new_repr(self)
    else:
        warnings.warn(_warning_msg, DeprecationWarning, stacklevel=2)
        return expr_repr(self)


def _warning_repr_html(self):
    if use_new_repr:
        return new_repr(self)
    else:
        warnings.warn(_warning_msg, DeprecationWarning, stacklevel=2)
        return to_html(self)


def new_repr(self):
    fmt =  "<`{}` expression; dshape='{}'>"
    return fmt.format(type(self).__name__,
                      sanitized_dshape(self.dshape))


Expr.__repr__ = _choose_repr
Expr.peek = _peek
Expr._repr_html_ = _warning_repr_html
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


CORE_SCALAR_TYPES = (float, decimal.Decimal, int, bool, str, Timestamp,
                     datetime.date, datetime.timedelta)
CORE_SEQUENCE_TYPES = (list, dict, tuple, set, Series, DataFrame, np.ndarray)
CORE_TYPES = CORE_SCALAR_TYPES + CORE_SEQUENCE_TYPES


def iscorescalar(x):
    return isinstance(x, CORE_SCALAR_TYPES)


def iscoresequence(x):
    return isinstance(x, CORE_SEQUENCE_TYPES)


def iscoretype(x):
    return isinstance(x, CORE_TYPES)


Expr.__array__ = intonumpy
Expr.__int__ = lambda x: convert_base(int, x)
Expr.__float__ = lambda x: convert_base(float, x)
Expr.__complex__ = lambda x: convert_base(complex, x)
Expr.__bool__ = lambda x: convert_base(bool, x)
Expr.__nonzero__ = lambda x: convert_base(bool, x)
Expr.__iter__ = into(Iterator)
