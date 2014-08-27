from __future__ import absolute_import, division, print_function

import os
from functools import partial
from dynd import nd
import datashape
import sys
from functools import partial
from datashape import dshape, Record, to_numpy_dtype, Option
from datashape.predicates import isscalar
import toolz
from toolz import concat, partition_all, valmap, first, merge
from cytoolz import pluck, compose
import copy
from datetime import datetime
from numbers import Number
from collections import Iterable, Iterator
import numpy as np
import pandas as pd
import tables as tb

from ..compute.chunks import ChunkIterator, chunks
from ..compatibility import map
from ..dispatch import dispatch
from .. import expr
from ..expr import Expr, Projection, Field, Symbol
from ..compute.core import compute
from ..resource import resource
from ..compatibility import _strtypes, map
from ..utils import keywords
from ..data.utils import sort_dtype_items
from ..pytables import PyTables
from ..compute.spark import RDD


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
    from pymongo.collection import Collection
except ImportError:
    Collection = type(None)


try:
    from ..data import DataDescriptor, CSV, JSON, JSON_Streaming, Excel, SQL
except ImportError:
    DataDescriptor = type(None)
    CSV = type(None)
    JSON = type(None)
    JSON_STREAMING = type(None)
    Excel = type(None)

try:
    from rethinkdb.ast import Table as RqlTable, RqlQuery
    from rethinkdb.net import Cursor as RqlCursor
    from blaze.compute.rethink import RTable
except ImportError:
    RqlTable = type(None)
    RqlQuery = type(None)
    RTable = type(None)


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
                               type(dict().items()),
                               pd.Series, np.record, np.void))
def into(a, b, **kwargs):
    return type(a)(b)


@dispatch(set, list)
def into(a, b, **kwargs):
    try:
        return set(b)
    except TypeError:
        return set(map(tuple, b))

@dispatch(dict, (list, tuple, set))
def into(a, b, **kwargs):
    return dict(b)

@dispatch((list, tuple, set), dict)
def into(a, b, **kwargs):
    return type(a)(map(type(a), sorted(b.items(), key=lambda x: x[0])))

@dispatch(nd.array, (Iterable, Number) + _strtypes)
def into(a, b, **kwargs):
    return nd.array(b, **kwargs)

@dispatch(nd.array, nd.array)
def into(a, b, **kwargs):
    return b

@dispatch(np.ndarray, np.ndarray)
def into(a, b, **kwargs):
    return b

@dispatch(list, nd.array)
def into(a, b, **kwargs):
    return nd.as_py(b, tuple=True)

@dispatch(tuple, nd.array)
def into(a, b, **kwargs):
    return tuple(nd.as_py(b, tuple=True))

@dispatch(np.ndarray, nd.array)
def into(a, b, **kwargs):
    return nd.as_numpy(b, allow_copy=True)


def dtype_from_tuple(t):
    dshape = discover(t)
    names = ['f%d' % i for i in range(len(t))]
    types = [x.measure.to_numpy_dtype() for x in dshape.measure.dshapes]
    return np.dtype(list(zip(names, types)))


@dispatch(np.ndarray, (Iterable, Iterator))
def into(a, b, **kwargs):
    b = iter(b)
    first = next(b)
    b = toolz.concat([[first], b])
    if isinstance(first, datetime):
        b = map(np.datetime64, b)
    if isinstance(first, (list, tuple)):
        return np.rec.fromrecords([tuple(x) for x in b],
                                  dtype=kwargs.pop('dtype',
                                                   dtype_from_tuple(first)),
                                  **kwargs)
    elif hasattr(first, 'values'):
        #detecting sqlalchemy.engine.result.RowProxy types and similar
        return np.asarray([tuple(x.values()) for x in b], **kwargs)
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
def into(a, b, **kwargs):
    if 'M8' in str(b.dtype) or 'datetime' in str(b.dtype):
        b = b.astype(degrade_numpy_dtype_to_python(b.dtype))
    return numpy_ensure_strings(b).tolist()


@dispatch(set, object)
def into(a, b, **kwargs):
    return set(into(list, b, **kwargs))


@dispatch(pd.DataFrame, np.ndarray)
def into(df, x, **kwargs):
    if len(df.columns) > 0:
        columns = list(df.columns)
    else:
        columns = list(x.dtype.names)
    return pd.DataFrame(numpy_ensure_strings(x), columns=columns)


@dispatch((pd.DataFrame, list, tuple, Iterator, nd.array), tb.Table)
def into(a, t, **kwargs):
    x = into(np.ndarray, t)
    return into(a, x, **kwargs)


@dispatch(np.ndarray, tb.Table)
def into(_, t, **kwargs):
    res = t[:]
    dt_fields = [k for k, v in t.coltypes.items() if v == 'time64']

    if not dt_fields:
        return res

    for f in dt_fields:
        # pytables is in seconds since epoch
        res[f] *= 1e6

    fields = []
    for name, dtype in sort_dtype_items(t.coldtypes.items(), t.colnames):
        typ = getattr(t.cols, name).type
        fields.append((name, {'time64': 'datetime64[us]',
                              'time32': 'datetime64[D]',
                              'string': dtype.str}.get(typ, typ)))
    return res.astype(np.dtype(fields))


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
        dt = [(n, "S%d" % max(map(len, x[n]))
               if x.dtype[n] == 'O' else x.dtype[n])
                for n in x.dtype.names]
        x = x.astype(dt)
    return x


def typehint(x, typedict):
    """Replace the dtypes in `x` keyed by `typedict` with the dtypes in
    `typedict`.
    """
    dtype = x.dtype
    lhs = dict(zip(dtype.fields.keys(), map(first, dtype.fields.values())))
    dtype_list = list(merge(lhs, typedict).items())
    return x.astype(np.dtype(sort_dtype_items(dtype_list, dtype.names)))


@dispatch(tb.Table, np.ndarray)
def into(t, x, **kwargs):
    dt_types = dict((k, 'datetime64[us]') for k, (v, _) in
                    x.dtype.fields.items() if issubclass(v.type, np.datetime64))
    x = numpy_ensure_bytes(numpy_fixlen_strings(x))
    x = typehint(typehint(x, dt_types), dict.fromkeys(dt_types, 'f8'))

    for name in dt_types:
        x[name] /= 1e6

    t.append(x)
    return t


@dispatch(tb.Table, ChunkIterator)
def into(t, c, **kwargs):
    for chunk in c:
        into(t, chunk, **kwargs)
    return t


@dispatch(tb.node.MetaNode, tb.Table)
def into(table, data, filename=None, datapath=None, **kwargs):
    dshape = datashape.dshape(kwargs.setdefault('dshape', discover(data)))
    t = PyTables(filename, datapath=datapath, dshape=dshape)
    return into(t, data)


@dispatch(ctable, tb.Table)
def into(bc, data, **kwargs):
    cs = chunks(data)
    bc = into(bc, next(cs))
    for chunk in cs:
        bc.append(chunk)
    return bc


@dispatch(tb.node.MetaNode, np.ndarray)
def into(_, x, filename=None, datapath=None, **kwargs):
    # tb.node.MetaNode == type(tb.Table)
    x = numpy_ensure_bytes(numpy_fixlen_strings(x))
    t = PyTables(filename, datapath=datapath, dshape=discover(x))
    return into(t, x, **kwargs)


@dispatch(tb.node.MetaNode, (ctable, list))
def into(_, data, filename=None, datapath=None, **kwargs):
    t = PyTables(filename, datapath=datapath,
                 dshape=kwargs.get('dshape', discover(data)))
    for chunk in map(partial(into, np.ndarray), chunks(data)):
        into(t, chunk)
    return t


@dispatch(tb.Table, (pd.DataFrame, CSV, SQL, nd.array, Collection))
def into(a, b, **kwargs):
    return into(a, into(np.ndarray, b), **kwargs)


@dispatch(tb.Table, _strtypes)
def into(a, b, **kwargs):
    kw = dict(kwargs)
    if 'output_path' in kw:
        del kw['output_path']
    r = resource(b, **kw)
    return into(a, r, **kwargs)


@dispatch(list, pd.DataFrame)
def into(_, df, **kwargs):
    return into([], into(np.ndarray(0), df))


@dispatch(pd.DataFrame, nd.array)
def into(a, b, **kwargs):
    ds = dshape(nd.dshape_of(b))
    if list(a.columns):
        names = list(a.columns)
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
def into(_, df, **kwargs):
    return df.copy()

@dispatch(pd.Series, pd.Series)
def into(_, ser, **kwargs):
    return ser

@dispatch(pd.Series, Iterator)
def into(a, b, **kwargs):
    return into(a, list(b), **kwargs)

@dispatch(pd.Series, (list, tuple))
def into(a, b, **kwargs):
    return pd.Series(b, **kwargs)

@dispatch(pd.Series, Expr)
def into(ser, col, **kwargs):
    ser = into(ser, compute(col))
    ser.name = col._name
    return ser


@dispatch(pd.Series, pd.DataFrame)
def into(a, b, **kwargs):
    if len(b.columns) != 1:
        raise TypeError('Cannot transform a multiple column expression to a'
                        ' Series')
    s = b.squeeze()
    if a.name is not None:
        s.name = a.name
    return s


@dispatch(pd.Series, Projection)
def into(ser, col, **kwargs):
    return into(pd.Series, into(pd.DataFrame, col))


@dispatch(pd.Series, np.ndarray)
def into(s, x, **kwargs):
    return pd.Series(numpy_ensure_strings(x), name=s.name)

@dispatch(pd.DataFrame, pd.Series)
def into(df, s, **kwargs):
    assert len(df.columns) <= 1, 'DataFrame columns must be empty or length 1'
    return pd.DataFrame(s, columns=df.columns if len(df.columns) else [s.name])


@dispatch(list, pd.Series)
def into(_, ser, **kwargs):
    return ser.tolist()

@dispatch(nd.array, pd.DataFrame)
def into(a, df, **kwargs):
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


@dispatch(pd.Series)
def discover(s):
    return discover(s.to_frame())


@dispatch(np.ndarray, carray)
def into(a, b, **kwargs):
    return b[:]

@dispatch(pd.Series, carray)
def into(a, b, **kwargs):
    return into(a, into(np.ndarray, b))

@dispatch(ColumnDataSource, (pd.DataFrame, np.ndarray, ctable))
def into(cds, t, **kwargs):
    columns = discover(t).subshape[0][0].names
    return ColumnDataSource(data=dict((col, into([], t[col]))
                                      for col in columns))

@dispatch(ColumnDataSource, Expr)
def into(cds, t, **kwargs):
    columns = t.fields
    return ColumnDataSource(data=dict((col, into([], t[col]))
                                      for col in columns))


@dispatch(ColumnDataSource, tb.Table)
def into(cds, t, **kwargs):
    return into(cds, into(pd.DataFrame, t))


@dispatch(ColumnDataSource, nd.array)
def into(cds, t, **kwargs):
    columns = discover(t).subshape[0][0].names
    return ColumnDataSource(data=dict((col, into([], getattr(t, col)))
                                      for col in columns))

@dispatch(ColumnDataSource, Collection)
def into(cds, other, **kwargs):
    return into(cds, into(pd.DataFrame, other))


@dispatch(ctable, Expr)
def into(a, b, **kwargs):
    c = compute(b)
    if isinstance(c, (list, tuple, Iterator)):
        kwargs['types'] = [datashape.to_numpy_dtype(t) for t in
                b.schema[0].types]
        kwargs['names'] = b.fields
    return into(a, c, **kwargs)


@dispatch(pd.DataFrame, ColumnDataSource)
def into(df, cds, **kwargs):
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
    kwargs = toolz.keyfilter(keywords(ctable).__contains__, kwargs)
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
    --------
    >>> x = np.array(['a', 'b'], dtype='S1')
    >>> # Python 2
    >>> numpy_ensure_strings(x)  # doctest: +SKIP
    np.array(['a', 'b'], dtype='S1')
    >>> # Python 3
    >>> numpy_ensure_strings(x)  # doctest: +SKIP
    np.array(['a', 'b'], dtype='U1')
    """
    if sys.version_info[0] >= 3 and 'S' in str(x.dtype):
        if x.dtype.names:
            dt = [(n, x.dtype[n].str.replace('S', 'U')) for n in x.dtype.names]
        else:
            dt = x.dtype.str.replace('S', 'U')
        x = x.astype(dt)
    return x


def numpy_ensure_bytes(x):
    """Return a numpy array whose string fields are converted to the bytes type
    appropriate for the Python version.

    Parameters
    ----------
    x : np.ndarray
        Record array

    Returns
    -------
    x : np.ndarray
        Record array with any unicode string type as a bytes type

    Examples
    --------
    >>> x = np.array(['a', 'b'])
    >>> # Python 2
    >>> numpy_ensure_bytes(x)  # doctest: +SKIP
    np.array(['a', 'b'], dtype='|S1')
    >>> # Python 3
    >>> numpy_ensure_strings(x)  # doctest: +SKIP
    np.array([b'a', b'b'], dtype='|S1')
    """
    if 'U' in str(x.dtype):
        if x.dtype.names is not None:
            dt = [(n, x.dtype[n].str.replace('U', 'S')) for n in x.dtype.names]
        else:
            dt = x.dtype.str.replace('U', 'S')
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


@dispatch(Collection, Expr)
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
    r = into(Iterator, coll, columns=columns, schema=schema)
    return type(l)(r)


@dispatch(Collection, CSV)
def into(coll, d, if_exists="replace", **kwargs):
    """
    Convert from TSV/CSV into MongoDB Collection

    Parameters
    ----------
    if_exists : string
        {replace, append, fail}
    header: bool (TSV/CSV only)
        Flag to define if file contains a header
    columns: list (TSV/CSV only)
        list of column names
    ignore_blank: bool
        Ignores empty fields in csv and tsv exports. Default: creates fields without values
    """
    import subprocess
    from dateutil import parser

    csv_dd = d
    db = coll.database

    copy_info = {
        'dbname': db.name,
        'coll': coll.name,
        'abspath': d._abspath
    }

    optional_flags = []

    if if_exists == 'replace':
        optional_flags.append('--drop')

    if kwargs.get('header', csv_dd.header):
        optional_flags.append('--headerline')
    if kwargs.get('ignore_blank', None):
        optional_flags.append('--ignoreBlanks')

    cols = kwargs.get('columns', csv_dd.columns)
    copy_info['column_names'] = ','.join(cols)

    delim = csv_dd.dialect['delimiter']
    typ = copy_info['file_type'] = {',': 'csv', '\t': 'tsv'}.get(delim, None)
    if typ is None:
        dd_into_coll = into.dispatch(Collection, DataDescriptor)
        return dd_into_coll(coll, csv_dd)

    copy_cmd = ("mongoimport -d {dbname} -c {coll} --type {file_type} "
                "--file {abspath} --fields {column_names} ")

    copy_cmd = copy_cmd.format(**copy_info) + ' '.join(optional_flags)

    ps = subprocess.Popen(copy_cmd, shell=os.name != 'nt',
                          stdout=subprocess.PIPE)
    ps.wait()

    # need to check for date columns and update
    date_cols = []
    dshape = csv_dd.dshape
    for t, c in zip(dshape[1].types, dshape[1].names):
        if isinstance(t, Option):
            t = t.ty
        if isinstance(t, (datashape.Date, datashape.DateTime)):
            date_cols.append((c, t))

    for d_col, ty in date_cols:
        mongo_data = list(coll.find({}, {d_col: 1}))
        for doc in mongo_data:
            try:
                t = parser.parse(doc[d_col])
            except AttributeError:
                t = doc[d_col]
            m_id = doc['_id']
            coll.update({'_id': m_id}, {"$set": {d_col: t}})

    return coll


@dispatch(Collection, (JSON, JSON_Streaming))
def into(coll, d, if_exists="replace", **kwargs):
    """
    into function which converts TSV/CSV/JSON into a MongoDB Collection
    Parameters
    ----------
    if_exists : string
        {replace, append, fail}
    json_array : bool
        Accepts the import of data expressed with multiple MongoDB documents within a single JSON array.
    """
    import subprocess

    json_dd = d
    db = coll.database

    copy_info = {
        'dbname': db.name,
        'coll': coll.name,
        'abspath': d._abspath
    }
    optional_flags = []

    if if_exists == 'replace':
        optional_flags.append('--drop')

    if kwargs.get('json_array', None):
        optional_flags.append('--jsonArray')

    copy_info['file_type'] = 'json'

    copy_cmd = ("mongoimport -d {dbname} -c {coll} --type {file_type} "
                "--file {abspath} ")

    copy_cmd = copy_cmd.format(**copy_info) + ' '.join(optional_flags)

    ps = subprocess.Popen(copy_cmd, shell=os.name != 'nt',
                          stdout=subprocess.PIPE)
    ps.wait()


@dispatch(nd.array, DataDescriptor)
def into(_, dd, **kwargs):
    return dd.dynd[:]


@dispatch(Iterator, DataDescriptor)
def into(_, dd, **kwargs):
    return iter(dd)


@dispatch((np.ndarray, ColumnDataSource, ctable), DataDescriptor)
def into(a, b, **kwargs):
    return into(a, into(nd.array(), b), **kwargs)


@dispatch((np.ndarray, ColumnDataSource, ctable, tb.Table, list, tuple, set),
          (CSV, Excel))
def into(a, b, **kwargs):
    return into(a, into(pd.DataFrame(), b, **kwargs), **kwargs)

@dispatch(ColumnDataSource, pd.Series)
def into(a, b, **kwargs):
    return ColumnDataSource(data={b.name: b.tolist()})


@dispatch((list, tuple, set), ColumnDataSource)
def into(a, cds, **kwargs):
    if not isinstance(a, type):
        a = type(a)
    return a(zip(*cds.data.values()))

@dispatch(pd.DataFrame, CSV)
def into(a, b, **kwargs):
    # Pass only keyword arguments appropriate for read_csv
    kws = keywords(pd.read_csv)
    options = toolz.merge(b.dialect, kwargs)
    options = toolz.keyfilter(kws.__contains__, options)
    return b.pandas_read_csv(chunksize=None, **options)


@dispatch((np.ndarray, pd.DataFrame, ColumnDataSource, ctable, tb.Table, list,
           tuple, set), (Projection, Field))
def into(a, b, **kwargs):
    """ Special case on anything <- Data(CSV)[columns]

    Many CSV injest functions have keyword arguments to take only certain
    columns.  We should leverage these if our input is of the form like the
    following for CSVs

    >>> csv = CSV('/path/to/file.csv')              # doctest: +SKIP
    >>> t = Data(csv)                               # doctest: +SKIP
    >>> into(list, t[['column-1', 'column-2']])     # doctest: +SKIP
    """
    if isinstance(b._child, Symbol) and isinstance(b._child.data, CSV):
        kwargs.setdefault('names', b._child.fields)
        kwargs.setdefault('usecols', b.fields)
        kwargs.setdefault('squeeze', isscalar(b.dshape.measure))
        return into(a, b._child.data, **kwargs)
    else:
        # TODO, replace with with raise MDNotImplementeError once
        # https://github.com/mrocklin/multipledispatch/pull/39 is merged
        a = a if isinstance(a, type) else type(a)
        f = into.dispatch(a, Expr)
        return f(a, b, **kwargs)

    # TODO: add signature for SQL import

    # TODO: CSV of Field




@dispatch(pd.DataFrame, DataDescriptor)
def into(a, b):
    return pd.DataFrame(list(b), columns=b.columns)


@dispatch(pd.DataFrame, Concat)
def into(a, b, **kwargs):
    """Convert a sequence of DataDescriptors to a DataFrame by converting each
    to a DataFrame and then calling pandas.concat on the resulting sequence.
    """
    return pd.concat((into(pd.DataFrame, d) for d in b.descriptors),
                     ignore_index=kwargs.pop('ignore_index', True),
                     **kwargs)


@dispatch(object, Expr)
def into(a, b):
    return compute(b)


@dispatch(_strtypes, _strtypes)
def into(a, b, **kwargs):
    """ Transfer data between two URIs

    Transfer data between two data resources based on their URIs.

    >>> into('sqlite://:memory:::tablename', '/path/to/file.csv') #doctest:+SKIP
    <blaze.data.sql.SQL at 0x7f32d80b80d0>

    Uses ``resource`` functin to resolve data resources

    See Also
    --------

    blaze.resource.resource
    """
    b = resource(b, **kwargs)
    return into(a, b, **kwargs)


@dispatch((type, RDD, set, np.ndarray, object), _strtypes)
def into(a, b, **kwargs):
    return into(a, resource(b, **kwargs), **kwargs)


@dispatch(_strtypes, (Expr, RDD, object))
def into(a, b, **kwargs):
    dshape = kwargs.pop('dshape', None)
    dshape = dshape or discover(b)
    if isinstance(dshape, str):
        dshape = datashape.dshape(dshape)
    target = resource(a, dshape=dshape,
                         schema=dshape.subshape[0],
                         mode='a',
                         **kwargs)
    return into(target, b, dshape=dshape, **kwargs)

@dispatch(Iterator, (list, tuple, set, Iterator))
def into(a, b):
    return b

@dispatch(pd.DataFrame, Excel)
def into(df, xl):
    return pd.read_excel(xl.path, sheetname=xl.worksheet)

@dispatch(pd.DataFrame, ChunkIterator)
def into(df, chunks, **kwargs):
    dfs = [into(df, chunk, **kwargs) for chunk in chunks]
    return pd.concat(dfs, ignore_index=True)


@dispatch(np.ndarray, ChunkIterator)
def into(x, chunks, **kwargs):
    arrs = [into(x, chunk, **kwargs) for chunk in chunks]
    return np.vstack(arrs)

@dispatch((DataDescriptor, Collection), ChunkIterator)
def into(coll, chunks, **kwargs):
    for chunk in chunks:
        into(coll, chunk, **kwargs)
    return coll


@dispatch((list, tuple, set), DataDescriptor)
def into(a, b, **kwargs):
    if not isinstance(a, type):
        a = type(a)
    return a(b)


@dispatch(DataDescriptor, (list, tuple, set, DataDescriptor, Iterator))
def into(a, b, **kwargs):
    a.extend(b)
    return a

@dispatch(DataDescriptor, (np.ndarray, nd.array, pd.DataFrame, Collection))
def into(a, b, **kwargs):
    a.extend(into(list,b))
    return a


@dispatch(Number, Number)
def into(a, b, **kwargs):
    if not isinstance(a, type):
        a = type(a)
    return a(b)


@dispatch(object)
def into(a, **kwargs):
    """ Curried into function

    >>> f = into(list)
    >>> f((1, 2, 3))
    [1, 2, 3]
    """
    def partial_into(b, **kwargs2):
        return into(a, b, **merge(kwargs, kwargs2))
    return partial_into


# This is only here due to a conflict
# Which is only because issubclass(carray, Iterable)
@dispatch(Collection, carray)
def into(a, b, **kwargs):
    into(a, into(Iterator, b, **kwargs))
    return a


@dispatch(RqlTable, (RqlTable, list))
def into(t, r):
    return t.insert(r)


@dispatch(RqlTable, RqlCursor)
def into(t, rc):
    return into(t, list(rc))


@dispatch(RqlTable, np.recarray)
def into(t, a):
    rec_list = list(map(partial(compose(dict, zip), a.dtype.fields.keys()), a))
    return into(t, rec_list)


@dispatch(RTable, RTable)
def into(r1, r2):
    return into(r1, r2.t)


@dispatch(RTable, (RqlTable, list, np.recarray, RqlCursor))
def into(r, o):
    into(r.t, o).run(r.conn)
    return r
