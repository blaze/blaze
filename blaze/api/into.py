from __future__ import absolute_import, division, print_function

from dynd import nd
import datashape
import sys
from datashape import DataShape, dshape, Record, to_numpy_dtype
import toolz
from toolz import concat, partition_all, valmap
from cytoolz import pluck
import copy
from datetime import datetime
from datashape.user import validate, issubschema
from numbers import Number
from collections import Iterable, Iterator
import gzip
import numpy as np
import pandas as pd
import h5py
import tables

from ..compute.chunks import ChunkIterator
from ..dispatch import dispatch
from ..expr import TableExpr, Expr
from ..compute.core import compute
from .resource import resource
from ..compatibility import _strtypes
from ..utils import keywords


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

    >>> into(pd.DataFrame, _)
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
    from ..data import DataDescriptor, CSV
except ImportError:
    DataDescriptor = type(None)
    CSV = type(None)


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
    f = into.dispatch(a, type(b))
    try:
        a = a()
    except:
        pass
    return f(a, b, **kwargs)

@dispatch((list, tuple, set), (list, tuple, set, Iterator,
                               type(dict().items())))
def into(a, b):
    return type(a)(b)


@dispatch(set, list)
def into(a, b):
    try:
        return set(b)
    except TypeError:
        return set(map(tuple, b))


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
def into(a, b, **kwargs):
    return b

@dispatch(list, nd.array)
def into(a, b):
    return nd.as_py(b, tuple=True)

@dispatch(tuple, nd.array)
def into(a, b):
    return tuple(nd.as_py(b, tuple=True))

@dispatch(np.ndarray, nd.array)
def into(a, b, **kwargs):
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

def degrade_numpy_dtype_to_python(dt):
    """

    >>> degrade_numpy_dtype_to_python(np.dtype('M8[ns]'))
    dtype('<M8[us]')
    >>> dt = np.dtype([('a', 'S7'), ('b', 'M8[D]'), ('c', 'M8[ns]')])
    >>> degrade_numpy_dtype_to_python(dt)
    dtype([('a', 'S7'), ('b', '<M8[D]'), ('c', '<M8[us]')])
    """
    replacements = {'M8[ns]': np.dtype('M8[us]'),
                    'M8[as]': np.dtype('M8[us]')}
    dt = replacements.get(dt.str.lstrip('<>'), dt)

    if str(dt)[0] == '[':
        return np.dtype([(name, degrade_numpy_dtype_to_python(dt[name]))
                        for name in dt.names])
    return dt


@dispatch(list, np.ndarray)
def into(a, b):
    if 'M8' in str(b.dtype) or 'datetime' in str(b.dtype):
        b = b.astype(degrade_numpy_dtype_to_python(b.dtype))
    return numpy_ensure_strings(b).tolist()


@dispatch(pd.DataFrame, np.ndarray)
def into(df, x):
    if len(df.columns) > 0:
        columns = list(df.columns)
    else:
        columns = list(x.dtype.names)
    return pd.DataFrame(numpy_ensure_strings(x), columns=columns)

@dispatch((pd.DataFrame, list, tuple, Iterator, nd.array), tables.Table)
def into(a, t):
    x = into(np.ndarray, t)
    return into(a, x)


@dispatch(np.ndarray, tables.Table)
def into(_, t):
    return t[:]


def numpy_fixlen_strings(x):
    """ Returns new array with strings as fixed length

    >>> from numpy import rec
    >>> x = rec.array([(1, 'Alice', 100), (2, 'Bob', 200)],
    ...               dtype=[('id', 'i8'), ('name', 'O'), ('amount', 'i8')])

    >>> numpy_fixlen_strings(x) # doctest: +SKIP
    rec.array([(1, 'Alice', 100), (2, 'Bob', 200)],
          dtype=[('id', '<i8'), ('name', 'S5'), ('amount', '<i8')])
    """
    if "'O'" in str(x.dtype):
        dt = [(n, "S%d" % max(map(len, x[n])) if x.dtype[n] == 'O' else x.dtype[n])
                for n in x.dtype.names]
        x = x.astype(dt)
    return x

@dispatch(tables.Table, np.ndarray)
def into(_, x, filename=None, datapath=None, **kwargs):
    if filename is None or datapath is None:
        raise ValueError("Must specify filename for new PyTables file. \n"
        "Example: into(tb.Tables, df, filename='myfile.h5', datapath='/data')")

    f = tables.open_file(filename, 'w')
    t = f.create_table('/', datapath, obj=numpy_fixlen_strings(x))
    return t


@dispatch(tables.Table, pd.DataFrame)
def into(a, df, **kwargs):
    return into(a, into(np.ndarray, df), **kwargs)
    # store = pd.HDFStore(filename, mode='w')
    # store.put(datapath, df, format='table', data_columns=True, index=False)
    # return getattr(store.root, datapath).table


@dispatch(tables.Table, _strtypes)
def into(a, b, **kwargs):
    kw = dict(kwargs)
    if 'output_path' in kw:
        del kw['output_path']
    return into(a, resource(b, **kw), **kwargs)


@dispatch(list, pd.DataFrame)
def into(_, df):
    return into([], into(np.ndarray(0), df))

@dispatch(pd.DataFrame, nd.array)
def into(a, b):
    ds = dshape(nd.dshape_of(b))
    if list(a.columns):
        names = a.columns
    elif isinstance(ds[-1], Record):
        names = ds[-1].names
    else:
        names = None
    if names:
        return pd.DataFrame(nd.as_py(b), columns=names)
    else:
        return pd.DataFrame(nd.as_py(b))

@dispatch(pd.DataFrame, (list, tuple, Iterator, type(dict().items())))
def into(df, seq, **kwargs):
    if list(df.columns):
        return pd.DataFrame(list(seq), columns=df.columns, **kwargs)
    else:
        return pd.DataFrame(list(seq), **kwargs)

@dispatch(pd.DataFrame, pd.DataFrame)
def into(_, df):
    return df.copy()

@dispatch(pd.Series, pd.Series)
def into(_, ser):
    return ser

@dispatch(pd.Series, Iterator)
def into(a, b, **kwargs):
    return into(a, list(b), **kwargs)

@dispatch(pd.Series, (list, tuple))
def into(a, b, **kwargs):
    return pd.Series(b, **kwargs)

@dispatch(pd.Series, TableExpr)
def into(ser, col):
    ser = into(ser, compute(col))
    ser.name = col.name
    return ser

@dispatch(pd.Series, np.ndarray)
def into(_, x):
    return pd.Series(numpy_ensure_strings(x))
    df = into(pd.DataFrame(), x)
    return df[df.columns[0]]

@dispatch(pd.DataFrame, pd.Series)
def into(_, df):
    return pd.DataFrame(df)

@dispatch(list, pd.Series)
def into(_, ser):
    return ser.tolist()

@dispatch(nd.array, pd.DataFrame)
def into(a, df):
    schema = discover(df)
    arr = nd.empty(str(schema))
    for i in range(len(df.columns)):
        arr[:, i] = np.asarray(df[df.columns[i]])
    return arr


@dispatch(np.ndarray, pd.DataFrame)
def into(a, df, **kwargs):
    return df.to_records(index=False)


@dispatch(nd.array)
def discover(arr):
    return dshape(nd.dshape_of(arr))


@dispatch(pd.DataFrame)
def discover(df):
    obj = datashape.coretypes.object_
    names = list(df.columns)
    dtypes = list(map(datashape.CType.from_numpy_dtype, df.dtypes))
    dtypes = [datashape.string if dt == obj else dt for dt in dtypes]
    schema = Record(list(zip(names, dtypes)))
    return len(df) * schema


@dispatch(np.ndarray, carray)
def into(a, b, **kwargs):
    return b[:]

@dispatch(pd.Series, carray)
def into(a, b):
    return into(a, into(np.ndarray, b))

@dispatch(ColumnDataSource, (TableExpr, pd.DataFrame, np.ndarray, ctable))
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
    return into(cds, into(pd.DataFrame(), other))


@dispatch(pd.DataFrame, ColumnDataSource)
def into(df, cds):
    return cds.to_df()


@dispatch(ctable, TableExpr)
def into(a, b, **kwargs):
    c = compute(b)
    if isinstance(c, (list, tuple, Iterator)):
        kwargs['types'] = [datashape.to_numpy_dtype(t) for t in
                b.schema[0].types]
        kwargs['names'] = b.columns
    return into(a, c, **kwargs)


@dispatch(pd.DataFrame, ColumnDataSource)
def into(df, cds):
    return cds.to_df()


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


@dispatch(ctable, pd.DataFrame)
def into(a, df, **kwargs):
    return ctable([fix_len_string_filter(df[c]) for c in df.columns],
                      names=list(df.columns), **kwargs)


@dispatch(pd.DataFrame, ctable)
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
def into(coll, dd, chunksize=1024, **kwargs):
    return into(coll, iter(dd), chunksize=chunksize, schema=dd.schema)


@dispatch(Collection, (tuple, list, Iterator))
def into(coll, seq, columns=None, schema=None, chunksize=1024, **kwargs):
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


def numpy_ensure_strings(x):
    """ Return a new array with strings that will be turned into the str type

    In Python 3 the 'S' numpy type results in ``bytes`` objects.  This coerces the
    numpy type to a form that will create ``str`` objects

    Examples
    ========

    >>> x = np.array(['a', 'b'], dtype='S1')
    >>> # Python 2
    >>> numpy_ensure_strings(x)  # doctest: +SKIP
    np.array(['a', 'b'], dtype='S1')
    >>> # Python 3
    >>> numpy_ensure_strings(x)  # doctest: +SKIP
    np.array(['a', 'b'], dtype='U1')
    """
    if sys.version_info[0] >= 3 and "S" in str(x.dtype):
        if x.dtype.names:
            dt = [(n, x.dtype[n].str.replace('S', 'U')) for n in x.dtype.names]
            x = x.astype(dt)
        else:
            dt = x.dtype.str.replace('S', 'U')
            x = x.astype(dt)
    return x


@dispatch(Collection, (nd.array, np.ndarray))
def into(coll, x, **kwargs):
    return into(coll, into(pd.DataFrame(), x), **kwargs)


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


@dispatch(Collection, pd.DataFrame)
def into(coll, df, **kwargs):
    return into(coll, into([], df), columns=list(df.columns), **kwargs)


@dispatch(Collection, TableExpr)
def into(coll, t, **kwargs):
    from blaze import compute
    result = compute(t)
    return into(coll, result, schema=t.schema, **kwargs)


@dispatch(pd.DataFrame, Collection)
def into(df, coll, **kwargs):
    seq = list(coll.find())
    for item in seq:
        del item['_id']
    return pd.DataFrame(seq, **kwargs)


@dispatch((nd.array, np.ndarray), Collection)
def into(x, coll, **kwargs):
    return into(x, into(pd.DataFrame(), coll), **kwargs)


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


@dispatch(nd.array, DataDescriptor)
def into(_, dd, **kwargs):
    return dd.dynd[:]


@dispatch(Iterator, DataDescriptor)
def into(_, dd, **kwargs):
    return iter(dd)


@dispatch((list, tuple, set), DataDescriptor)
def into(c, dd, **kwargs):
    return type(c)(dd)


@dispatch((np.ndarray, pd.DataFrame, ColumnDataSource, ctable), DataDescriptor)
def into(a, b, **kwargs):
    return into(a, into(nd.array(), b), **kwargs)


@dispatch((np.ndarray, pd.DataFrame, ColumnDataSource, ctable, tables.Table,
    list, tuple, set),
          CSV)
def into(a, b, **kwargs):
    return into(a, into(pd.DataFrame(), b, **kwargs), **kwargs)


@dispatch(np.ndarray, CSV)
def into(a, b, **kwargs):
    return into(a, into(pd.DataFrame(), b, **kwargs))


@dispatch(pd.DataFrame, CSV)
def into(a, b, **kwargs):
    dialect = b.dialect.copy()
    del dialect['lineterminator']
    dates = [i for i, typ in enumerate(b.schema[0].types)
               if 'date' in str(typ)]
    schema = b.schema
    if '?' in str(schema):
        schema = dshape(str(schema).replace('?', ''))

    dtypes = valmap(to_numpy_dtype, schema[0].dict)

    datenames = [name for name in dtypes
                      if np.issubdtype(dtypes[name], np.datetime64)]

    dtypes = dict((k, v) for k, v in dtypes.items()
                         if not np.issubdtype(v, np.datetime64))

    if 'strict' in dialect:
        del dialect['strict']

    # Pass only keyword arguments appropriate for read_csv
    kws = keywords(pd.read_csv)
    options = toolz.merge(dialect, kwargs)
    options = toolz.keyfilter(lambda k: k in kws, options)

    if b.open == gzip.open:
        options['compression'] = 'gzip'

    return pd.read_csv(b.path,
                       skiprows=1 if b.header else 0,
                       dtype=dtypes,
                       parse_dates=datenames,
                       names=b.columns,
                       **options)


@dispatch(pd.DataFrame, DataDescriptor)
def into(a, b):
    return pd.DataFrame(list(b), columns=b.columns)


@dispatch(object, Expr)
def into(a, b):
    return compute(b)


@dispatch((tuple, list, Iterator, np.ndarray, pd.DataFrame, Collection, set,
    ctable), _strtypes)
def into(a, b, **kwargs):
    return into(a, resource(b, **kwargs), **kwargs)


@dispatch(Iterator, (list, tuple, set, Iterator))
def into(a, b):
    return b


@dispatch(pd.DataFrame, ChunkIterator)
def into(df, chunks, **kwargs):
    dfs = [into(df, chunk, **kwargs) for chunk in chunks]
    return pd.concat(dfs, ignore_index=True)


@dispatch(np.ndarray, ChunkIterator)
def into(x, chunks, **kwargs):
    arrs = [into(x, chunk, **kwargs) for chunk in chunks]
    return np.vstack(arrs)

@dispatch(Collection, ChunkIterator)
def into(coll, chunks, **kwargs):
    for chunk in chunks:
        into(coll, chunk, **kwargs)
    return coll
