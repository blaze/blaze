from __future__ import absolute_import, division, print_function

from dynd import nd
import datashape
from datashape import DataShape, dshape, Record
from datashape.user import validate, issubschema
from numbers import Number
from collections import Iterable, Iterator
import numpy as np

from ..dispatch import dispatch


__all__ = ['into', 'discover']


@dispatch(type, object)
def into(a, b):
    f = into.resolve((a, type(b)))
    try:
        a = a()
    except:
        pass
    return f(a, b)

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

@dispatch(np.ndarray, nd.array)
def into(a, b):
    return nd.as_numpy(b, allow_copy=True)

@dispatch(np.ndarray, Iterable)
def into(a, b):
    return np.asarray(b)

@dispatch(list, np.ndarray)
def into(a, b):
    return b.tolist()

from blaze.data import DataDescriptor
from pandas import DataFrame
@dispatch(DataFrame, DataDescriptor)
def into(a, b):
    return DataFrame(list(b), columns=b.columns)


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


@dispatch(nd.array, DataFrame)
def into(a, df):
    schema = discover(df)
    arr = nd.empty(str(schema))
    for i in range(len(df.columns)):
        arr[:, i] = np.asarray(df[df.columns[i]])
    return arr


@dispatch(DataFrame)
def discover(df):
    obj = datashape.coretypes.object_
    names = list(df.columns)
    dtypes = list(map(datashape.CType.from_numpy_dtype, df.dtypes))
    dtypes = [datashape.string if dt == obj else dt for dt in dtypes]
    schema = Record(list(zip(names, dtypes)))
    return len(df) * schema
