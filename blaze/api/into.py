from __future__ import absolute_import, division, print_function

from dynd import nd
import datashape
from datashape import DataShape, dshape, Record
import toolz
from datetime import datetime
from datashape.user import validate, issubschema
from numbers import Number
from collections import Iterable, Iterator
import numpy as np
from pandas import DataFrame, Series

from ..dispatch import dispatch
from ..expr.table import TableExpr
from ..compute.core import compute


__all__ = ['into', 'discover']


@dispatch(object, object)
def into(a, b, **kwargs):
    raise NotImplementedError(
            "Blaze does not know a rule for the following conversion"
            "\n%s <- %s" % (type(a).__name__, type(b).__name__))

# Optional imports

try:
    from bokeh.objects import ColumnDataSource
except ImportError:
    ColumnDataSource = type(None)

try:
    import bcolz
    from bcolz import ctable, carray
except ImportError:
    ctable = type(None)

try:
    from ..data.core import DataDescriptor
except ImportError:
    DataDescriptor = type(None)


@dispatch(type, object)
def into(a, b, **kwargs):
    """
    Resolve into when given a type as a first argument

    Usually we give into an example of the thing that we want

    >>> into([], (1, 2, 3)) # give me a list like []
    [1, 2, 3]

    However sometimes it's inconvenient to construct a dummy example.
    In those cases we just specify the desired type

    >>> into(list, (1, 2, 3))
    [1, 2, 3]
    """
    f = into.resolve((a, type(b)))
    try:
        a = a()
    except:
        pass
    return f(a, b, **kwargs)

@dispatch((list, tuple, set), (list, tuple, set, Iterator,
                               type(dict().items())))
def into(a, b):
    return type(a)(b)

@dispatch(dict, (list, tuple, set))
def into(a, b):
    return dict(b)

@dispatch((list, tuple, set), dict)
def into(a, b):
    return type(a)(map(type(a), sorted(b.items(), key=lambda x: x[0])))

@dispatch(nd.array, (Iterable, Number, str))
def into(a, b, **kwargs):
    return nd.array(b, **kwargs)

@dispatch(nd.array, nd.array)
def into(a, b):
    return b

@dispatch(np.ndarray, np.ndarray)
def into(a, b):
    return b

@dispatch(list, nd.array)
def into(a, b):
    return nd.as_py(b, tuple=True)

@dispatch(tuple, nd.array)
def into(a, b):
    return tuple(nd.as_py(b, tuple=True))

@dispatch(np.ndarray, nd.array)
def into(a, b):
    return nd.as_numpy(b, allow_copy=True)

@dispatch(np.ndarray, (Iterable, Iterator))
def into(a, b, **kwargs):
    b = iter(b)
    first = next(b)
    b = toolz.concat([[first], b])
    if isinstance(first, datetime):
        b = map(np.datetime64, b)
    if isinstance(first, (list, tuple)):
        return np.rec.fromrecords(list(b), **kwargs)
    else:
        return np.asarray(list(b), **kwargs)

@dispatch(list, np.ndarray)
def into(a, b):
    return b.tolist()


@dispatch(DataFrame, DataDescriptor)
def into(a, b):
    return DataFrame(list(b), columns=b.columns)


@dispatch(DataFrame, np.ndarray)
def into(df, x):
    if len(df.columns) > 0:
        columns = list(df.columns)
    else:
        columns = list(x.dtype.names)
    return DataFrame(x, columns=columns)

@dispatch(list, DataFrame)
def into(_, df):
    return np.asarray(df).tolist()

@dispatch(DataFrame, nd.array)
def into(a, b):
    ds = dshape(nd.dshape_of(b))
    if list(a.columns):
        names = a.columns
    elif isinstance(ds[-1], Record):
        names = ds[-1].names
    else:
        names = None
    if names:
        return DataFrame(nd.as_py(b), columns=names)
    else:
        return DataFrame(nd.as_py(b))

@dispatch(DataFrame, (list, tuple))
def into(df, seq, **kwargs):
    if list(df.columns):
        return DataFrame(list(seq), columns=df.columns, **kwargs)
    else:
        return DataFrame(list(seq), **kwargs)

@dispatch(DataFrame, DataFrame)
def into(_, df):
    return df.copy()

@dispatch(DataFrame, Series)
def into(_, df):
    return DataFrame(df)

@dispatch(nd.array, DataFrame)
def into(a, df):
    schema = discover(df)
    arr = nd.empty(str(schema))
    for i in range(len(df.columns)):
        arr[:, i] = np.asarray(df[df.columns[i]])
    return arr


@dispatch(np.ndarray, DataFrame)
def into(a, df):
    return df.to_records(index=False)


@dispatch(nd.array)
def discover(arr):
    return dshape(nd.dshape_of(arr))


@dispatch(DataFrame)
def discover(df):
    obj = datashape.coretypes.object_
    names = list(df.columns)
    dtypes = list(map(datashape.CType.from_numpy_dtype, df.dtypes))
    dtypes = [datashape.string if dt == obj else dt for dt in dtypes]
    schema = Record(list(zip(names, dtypes)))
    return len(df) * schema


assert TableExpr is not type(None)
assert ColumnDataSource is not type(None)
@dispatch(ColumnDataSource, (TableExpr, DataFrame))
def into(cds, t):
    return ColumnDataSource(data=dict((col, into(np.ndarray, t[col]))
                                      for col in t.columns))

@dispatch(DataFrame, ColumnDataSource)
def into(df, cds):
    return cds.to_df()

@dispatch(ctable, TableExpr)
def into(a, b, **kwargs):
    c = compute(b)
    if isinstance(c, (list, tuple)):
        kwargs['types'] = [datashape.to_numpy_dtype(t) for t in
                b.schema[0].types]
    return into(a, c, **kwargs)

@dispatch(ctable, nd.array)
def into(a, b, **kwargs):
    names = dshape(nd.dshape_of(b))[1].names
    columns = [into(np.ndarray(0), getattr(b, name)) for name in names]

    return bcolz.ctable(columns, names=names, **kwargs)
