from __future__ import absolute_import, division, print_function

from dynd import nd
import datashape
from datashape import DataShape, dshape, Record, to_numpy_dtype
import toolz
from toolz import concat, partition_all
from cytoolz import pluck
import copy
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
    """
    Push data in ``b`` into a container of type ``a``

    Examples
    --------

    >>> into([], (1, 2, 3))
    [1, 2, 3]

    >>> into(np.ndarray, [['Alice', 100], ['Bob', 200]], names=['name', 'amt'])
    rec.array([('Alice', 100), ('Bob', 200)],
                  dtype=[('name', 'S5'), ('amt', '<i8')])

    >>> into(DataFrame, _)
        name  amt
    0  Alice  100
    1    Bob  200
    """
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
    carray = type(None)

try:
    import pymongo
    from pymongo.collection import Collection
except ImportError:
    Collection = type(None)

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

@dispatch(list, Series)
def into(_, ser):
    return ser.tolist()

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


@dispatch(ColumnDataSource, (TableExpr, DataFrame, np.ndarray, ctable))
def into(cds, t):
    columns = discover(t).subshape[0][0].names
    return ColumnDataSource(data=dict((col, into([], t[col]))
                                      for col in columns))

@dispatch(ColumnDataSource, nd.array)
def into(cds, t):
    columns = discover(t).subshape[0][0].names
    return ColumnDataSource(data=dict((col, into([], getattr(t, col)))
                                      for col in columns))

@dispatch(ColumnDataSource, Collection)
def into(cds, other):
    return into(cds, into(DataFrame(), other))


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


def fix_len_string_filter(ser):
    """ Convert object strings to fixed length, pass through others """
    if ser.dtype == np.dtype('O'):
        return np.asarray(list(ser))
    else:
        return np.asarray(ser)


@dispatch(ctable, nd.array)
def into(a, b, **kwargs):
    names = dshape(nd.dshape_of(b))[1].names
    columns = [getattr(b, name) for name in names]
    columns = [np.asarray(nd.as_py(c))
            if to_numpy_dtype(dshape(nd.dshape_of(c))) == np.dtype('O')
            else into(np.ndarray(0), c) for c in columns]

    return bcolz.ctable(columns, names=names, **kwargs)



@dispatch(ctable, DataFrame)
def into(a, df, **kwargs):
    return ctable([fix_len_string_filter(df[c]) for c in df.columns],
                      names=list(df.columns), **kwargs)


@dispatch(DataFrame, ctable)
def into(a, b, **kwargs):
    return b.todataframe()


@dispatch(nd.array, ctable)
def into(a, b, **kwargs):
    return into(a, b[:], **kwargs)


@dispatch(ctable, ctable)
def into(a, b, **kwargs):
    if not kwargs and a == ctable:
        return b
    else:
        raise NotImplementedError()


@dispatch(Collection, DataDescriptor)
def into(coll, dd, chunksize=1024):
    return into(coll, iter(dd), chunksize=chunksize, schema=dd.schema)


@dispatch(Collection, (tuple, list, Iterator))
def into(coll, seq, columns=None, schema=None, chunksize=1024):
    seq = iter(seq)
    item = next(seq)
    seq = concat([[item], seq])

    if isinstance(item, (tuple, list)):
        if not columns and schema:
            columns = dshape(schema)[0].names
        if not columns:
            raise ValueError("Inputs must be dictionaries. "
                "Or provide columns=[...] or schema=DataShape(...) keyword")
        seq = (dict(zip(columns, item)) for item in seq)

    for block in partition_all(1024, seq):
        coll.insert(copy.deepcopy(block))

    return coll


@dispatch(Collection, (nd.array, np.ndarray))
def into(coll, x, **kwargs):
    return into(coll, into(DataFrame(), x), **kwargs)


@dispatch(Collection, ctable)
def into(coll, x, **kwargs):
    from blaze.bcolz import chunks
    for chunk in chunks(x):
        into(coll, chunk)


@dispatch(Collection, Collection)
def into(a, b, **kwargs):
    """ Copy collection on server-side

    https://groups.google.com/forum/#!topic/mongodb-user/wHqJFp44baY
    """
    b.database.command('eval', 'db.%s.copyTo("%s")' % (b.name, a.name),
                 nolock=True)
    return b


@dispatch(Collection, DataFrame)
def into(coll, df, **kwargs):
    return into(coll, into([], df), columns=list(df.columns), **kwargs)


@dispatch(Collection, TableExpr)
def into(coll, t, **kwargs):
    from blaze import compute
    result = compute(t)
    return into(coll, result, schema=t.schema, **kwargs)


@dispatch(DataFrame, Collection)
def into(df, coll, **kwargs):
    seq = list(coll.find())
    for item in seq:
        del item['_id']
    return DataFrame(seq, **kwargs)


@dispatch((nd.array, np.ndarray), Collection)
def into(x, coll, **kwargs):
    return into(x, into(DataFrame(), coll), **kwargs)


def _into_iter_mongodb(l, coll, columns=None, schema=None):
    """ Into helper function

    Return both a lazy sequence of tuples and a list of column names
    """
    seq = coll.find()
    if not columns and schema:
        columns = schema[0].names
    elif not columns:
        item = next(seq)
        seq = concat([[item], seq])
        columns = sorted(item.keys())
        columns.remove('_id')
    return columns, pluck(columns, seq)


@dispatch((carray, ctable), Collection)
def into(x, coll, columns=None, schema=None, **kwargs):
    columns, seq = _into_iter_mongodb(x, coll, columns=None, schema=None)
    return into(x, seq, names=columns, **kwargs)


@dispatch(Iterator, Collection)
def into(l, coll, columns=None, schema=None):
    columns, seq = _into_iter_mongodb(l, coll, columns=columns, schema=schema)
    return seq


@dispatch((tuple, list), Collection)
def into(l, coll, columns=None, schema=None):
    return type(l)(into(Iterator, coll, columns=columns, schema=schema))


