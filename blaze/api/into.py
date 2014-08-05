from __future__ import absolute_import, division, print_function

from dynd import nd
import datashape
from datashape import DataShape, dshape, Record
from datashape.user import validate, issubschema
from numbers import Number
from collections import Iterable, Iterator
import numpy as np
from pandas import DataFrame, Series

from ..dispatch import dispatch


__all__ = ['into', 'discover']


@dispatch(type, object)
def into(a, b, **kwargs):
    f = into.resolve((a, type(b)))
    try:
        a = a()
    except:
        pass
    return f(a, b, **kwargs)

@dispatch((list, tuple, set), (list, tuple, set, Iterator))
def into(a, b):
    return type(a)(b)

@dispatch(dict, (list, tuple, set))
def into(a, b):
    return dict(b)

@dispatch((list, tuple, set), dict)
def into(a, b):
    return type(a)(map(type(a), sorted(b.items(), key=lambda x: x[0])))

@dispatch(nd.array, (Iterable, Number, str))
def into(a, b):
    return nd.array(b)

@dispatch(list, nd.array)
def into(a, b):
    return nd.as_py(b)

@dispatch(tuple, nd.array)
def into(a, b):
    return tuple(nd.as_py(b, tuple=True))

@dispatch(np.ndarray, np.ndarray)
def into(a, b):
    return b

@dispatch(np.ndarray, nd.array)
def into(a, b):
    return nd.as_numpy(b, allow_copy=True)

@dispatch(np.ndarray, Iterator)
def into(a, b):
    return np.asarray(list(b))

@dispatch(np.ndarray, Iterable)
def into(a, b):
    return np.asarray(b)

@dispatch(list, np.ndarray)
def into(a, b):
    return b.tolist()

from blaze.data import DataDescriptor
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
def into(df, seq):
    if list(df.columns):
        return DataFrame(list(seq), columns=df.columns)
    else:
        return DataFrame(list(seq))

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


try:
    from bokeh.objects import ColumnDataSource
    from blaze.expr.table import TableExpr

    @dispatch(ColumnDataSource, (TableExpr, DataFrame))
    def into(cds, t):
        return ColumnDataSource(data=dict((col, into(np.ndarray, t[col]))
                                          for col in t.columns))

    @dispatch(DataFrame, ColumnDataSource)
    def into(df, cds):
        return cds.to_df()

except ImportError:
    pass


try:
    import h5py
    from ..compute.chunks import Chunks
    import bcolz
    import blz

    @dispatch((bcolz.ctable, bcolz.carray, blz.btable, blz.barray),
              (bcolz.ctable, bcolz.carray, blz.btable, blz.barray, h5py.Dataset))
    def into(a, b, **kwargs):
        return into(a, Chunks(b), **kwargs)

except ImportError:
    pass

