from __future__ import absolute_import, division, print_function

import sys
import itertools as it
import os
import gzip
from functools import partial

from multipledispatch import dispatch
from cytoolz import partition_all, merge, keyfilter, compose, first, second

import numpy as np
import pandas as pd
from datashape.discovery import discover, null, unpack
from datashape import dshape, Record, Option, Fixed, CType, Tuple, string
import datashape as ds

import blaze as bz
from .core import DataDescriptor
from ..api.resource import resource
from ..utils import nth, nth_list, keywords
from .. import compatibility
from ..compatibility import map, zip, PY2
from .utils import ordered_index, listpack

import csv

__all__ = ['CSV', 'drop']


numtypes = frozenset(ds.integral.types) | frozenset(ds.floating.types)
na_values = frozenset(filter(None, pd.io.parsers._NA_VALUES))


read_csv_kwargs = set(keywords(pd.read_csv))
assert read_csv_kwargs

to_csv_kwargs = set(keywords(pd.core.format.CSVFormatter.__init__))
assert to_csv_kwargs


def has_header(sample, encoding=sys.getdefaultencoding()):
    """Check whether a piece of sample text from a file has a header

    Parameters
    ----------
    sample : str
        Text to check for existence of a header
    encoding : str
        Encoding to use if ``isinstance(sample, bytes)``

    Returns
    -------
    h : bool or NoneType
        None if an error is thrown, otherwise ``True`` if a header exists and
        ``False`` otherwise.
    """
    sniffer = csv.Sniffer().has_header

    try:
        return sniffer(sample)
    except TypeError:
        return sniffer(sample.decode(encoding))
    except csv.Error:
        return None


def get_dialect(sample, dialect=None, **kwargs):
    if isinstance(dialect, compatibility._strtypes):
        dialect = csv.get_dialect(dialect)

    sniffer = csv.Sniffer()
    if not dialect:
        try:
            dialect = sniffer.sniff(sample)
        except:
            dialect = csv.get_dialect('excel')

    # Convert dialect to dictionary
    dialect = dict((key, getattr(dialect, key))
                   for key in dir(dialect) if not key.startswith('_'))

    # Update dialect with any keyword arguments passed in
    # E.g. allow user to override with delimiter=','
    for k, v in kwargs.items():
        if k in dialect:
            dialect[k] = v

    return dialect


def discover_dialect(sample, dialect=None, **kwargs):
    """Discover a CSV dialect from string sample and additional keyword
    arguments

    Parameters
    ----------
    sample : str
    dialect : str or csv.Dialect

    Returns
    -------
    dialect : dict
    """
    dialect = get_dialect(sample, dialect, **kwargs)
    assert dialect

    # Pandas uses sep instead of delimiter.
    # Lets support that too
    if 'sep' in kwargs:
        dialect['delimiter'] = kwargs['sep']
    else:
        # but only on read_csv, to_csv doesn't accept delimiter so we need sep
        # for sure
        dialect['sep'] = dialect['delimiter']

    # pandas doesn't like two character newline terminators and line_terminator
    # is for to_csv
    dialect['lineterminator'] = dialect['line_terminator'] = \
        dialect['lineterminator'].replace('\r\n', '\n').replace('\r', '\n')

    return dialect


def get_sample(csv, size=16384):
    if os.path.exists(csv.path) and csv.mode != 'w':
        f = csv.open(csv.path)
        try:
            return f.read(size)
        finally:
            try:
                f.close()
            except AttributeError:
                pass
    return ''


def isdatelike(typ):
    return (typ == ds.date_ or typ == ds.datetime_ or
            (isinstance(typ, Option) and
             (typ.ty == ds.date_ or typ.ty == ds.datetime_)))


def get_date_columns(schema):
    try:
        names = schema.measure.names
        types = schema.measure.types
    except AttributeError:
        return []
    else:
        return [(name, typ) for name, typ in zip(names, types)
                if isdatelike(typ)]


def get_pandas_dtype(typ):
    # ugh conform to pandas "everything empty is a float or object",
    # otherwise we get '' trying to be an integer
    if isinstance(typ, Option):
        if typ.ty in numtypes:
            return np.dtype('f8')
        return typ.ty.to_numpy_dtype()
    return typ.to_numpy_dtype()


class CSV(DataDescriptor):
    """
    Blaze data descriptor to a CSV file.

    This reads in a portion of the file to discover the CSV dialect
    (i.e delimiter, endline character, ...), the column names (from the header)
    and the types (by looking at the values in the first 50 lines.  Often this
    just works however for complex datasets you may have to supply more
    metadata about your file.

    For full automatic handling just specify the filename

    >>> dd = CSV('myfile.csv')  # doctest: +SKIP

    Standard csv parsing terms like ``delimiter`` are available as keyword
    arguments.  See the standard ``csv`` library for more details on dialects.

    >>> dd = CSV('myfile.csv', delimiter='\t') # doctest: +SKIP

    If column names are not present in the header, specify them with the
    columns keyword argument

    >>> dd = CSV('myfile.csv',
    ...          columns=['id', 'name', 'timestamp', 'value'])  # doctest: +SKIP

    If a few types are not correctly discovered from the data then add additional
    type hints.

    >>> dd = CSV('myfile.csv',
    ...          columns=['id', 'name', 'timestamp', 'value'],
    ...          typehints={'timestamp': 'datetime'}) # doctest: +SKIP

    Alternatively specify all types manually

    >>> dd = CSV('myfile.csv',
    ...          columns=['id', 'name', 'timestamp', 'value'],
    ...          types=['int', 'string', 'datetime', 'float64'])  # doctest: +SKIP

    Or specify a datashape explicitly

    >>> schema = '{id: int, name: string, timestamp: datetime, value: float64}'
    >>> dd = CSV('myfile.csv', schema=schema)  # doctest: +SKIP

    Parameters
    ----------
    path : string
        A path string for the CSV file.
    schema : string or datashape
        A datashape (or its string representation) of the schema
        in the CSV file.
    dialect : string or csv.Dialect instance
        The dialect as understood by the `csv` module in Python standard
        library.  If not specified, a value is guessed.
    header : boolean
        Whether the CSV file has a header or not.  If not specified a value
        is guessed.
    open : context manager
        An alternative method to open the file.
        For examples: gzip.open, codecs.open
    nrows_discovery : int
        Number of rows to read when determining datashape
    """
    def __init__(self, path, mode='rt', schema=None, columns=None, types=None,
                 typehints=None, dialect=None, header=None, open=open,
                 nrows_discovery=50, chunksize=1024,
                 encoding=sys.getdefaultencoding(), **kwargs):
        if 'r' in mode and not os.path.isfile(path):
            raise ValueError('CSV file "%s" does not exist' % path)

        if schema is None and 'w' in mode:
            raise ValueError('Please specify schema for writable CSV file')

        self.path = path
        self.mode = mode
        self.open = open
        self.header = header
        self._abspath = os.path.abspath(path)
        self.chunksize = chunksize
        self.encoding = encoding

        sample = get_sample(self)
        self.dialect = dialect = discover_dialect(sample, dialect, **kwargs)

        if header is None:
            header = has_header(sample, encoding=encoding)
        elif isinstance(header, int):
            dialect['header'] = header
            header = True

        reader_dialect = keyfilter(read_csv_kwargs.__contains__, dialect)
        if not schema and 'w' not in mode:
            if not types:
                data = self.reader(skiprows=1, nrows=nrows_discovery,
                                   as_recarray=True, index_col=False,
                                   **reader_dialect).tolist()
                types = discover(data)
                rowtype = types.subshape[0]
                if isinstance(rowtype[0], Tuple):
                    types = types.subshape[0][0].dshapes
                    types = [unpack(t) for t in types]
                    types = [string if t == null else t for t in types]
                    types = [t if isinstance(t, Option) or t == string
                             else Option(t) for t in types]
                elif (isinstance(rowtype[0], Fixed) and
                        isinstance(rowtype[1], CType)):
                    types = int(rowtype[0]) * [rowtype[1]]
                else:
                    raise ValueError("Could not discover schema from data.\n"
                                     "Please specify schema.")
            if not columns:
                if header:
                    columns = first(self.reader(skiprows=0, nrows=1,
                                                header=None, **reader_dialect
                                                ).itertuples(index=False))
                else:
                    columns = ['_%d' % i for i in range(len(types))]
            if typehints:
                types = [typehints.get(c, t) for c, t in zip(columns, types)]

            schema = dshape(Record(list(zip(columns, types))))

        self._schema = schema
        self.header = header

    def reader(self, header=None, keep_default_na=False,
               na_values=na_values, chunksize=None, **kwargs):
        kwargs.setdefault('skiprows', int(bool(self.header)))

        dialect = merge(keyfilter(read_csv_kwargs.__contains__, self.dialect),
                        kwargs)
        filename, ext = os.path.splitext(self.path)
        ext = ext.lstrip('.')
        reader = pd.read_csv(self.path, compression={'gz': 'gzip',
                                                     'bz2': 'bz2'}.get(ext),
                             chunksize=chunksize, na_values=na_values,
                             keep_default_na=keep_default_na,
                             encoding=self.encoding, header=header, **dialect)
        return reader

    def get_py(self, key):
        return self._get_py(ordered_index(key, self.dshape))

    def _get_py(self, key):
        if isinstance(key, tuple):
            assert len(key) == 2
            rows, cols = key
            usecols = ordered_index(cols, self.schema)
            usecols = None if isinstance(usecols, slice) else listpack(usecols)
        else:
            rows = key
            usecols = None

        reader = self._iter(usecols=usecols)
        if isinstance(rows, compatibility._inttypes):
            line = nth(rows, reader)
            try:
                return next(line).item()
            except TypeError:
                try:
                    return line.item()
                except AttributeError:
                    return line
        elif isinstance(rows, list):
            return nth_list(rows, reader)
        elif isinstance(rows, slice):
            return it.islice(reader, rows.start, rows.stop, rows.step)
        else:
            raise IndexError("key %r is not valid" % rows)


    def get_streaming_dtype(self, dtype):
        if not isinstance(self.schema.measure, Record):
            return dtype

        names = dtype.names
        types_names = ((i, t) for i, t in enumerate(self.schema.measure.types)
                       if str(i) in names)
        newtypes = [(str(i), get_pandas_dtype(typ)) for i, typ in types_names]

        # we only keep those fields that are in dtype
        formats = list(map(second, sorted(newtypes,
                                          key=lambda x: names.index(x[0]))))
        return np.dtype({'names': names, 'formats': formats})

    def _iter(self, usecols=None):
        # get the date column [(name, type)] pairs
        datecols = list(map(first, get_date_columns(self.schema)))

        # figure out which ones pandas needs to parse
        parse_dates = ordered_index(datecols, self.schema)
        if usecols is not None:
            parse_dates = [d for d in parse_dates if d in set(usecols)]

        reader = self.reader(chunksize=self.chunksize, parse_dates=parse_dates,
                             usecols=usecols, squeeze=True)

        # pop one off the iterator
        initial = next(iter(reader))
        if isinstance(initial, pd.Series):
            names = [str(initial.name)]
            formats = [initial.dtype]
            slicer = slice(None),
        else:
            if usecols is None:
                usecols = slice(None)
            names = list(map(str, initial.loc[:, usecols].columns))
            formats = initial.loc[:, usecols].dtypes.tolist()
            slicer = slice(None), usecols

        initial_dtype = np.dtype({'names': names, 'formats': formats})

        # what dtype do we actually want to see when we read
        streaming_dtype = self.get_streaming_dtype(initial_dtype)

        # everything must ultimately be a list
        m = partial(bz.into, list)
        slicerf = lambda x: x.loc[slicer].fillna('').convert_objects(
            convert_numeric=True)

        if streaming_dtype != initial_dtype:
            # we don't have the desired type so jump through hoops with
            # to_records -> astype(desired dtype) -> listify
            def mapper(x, dtype=streaming_dtype):
                r = slicerf(x)

                try:
                    r = r.to_records(index=False)
                except AttributeError:
                    # series doesn't allow struct dtypes
                    r = r.values
                    dtype = first(first(dtype.fields.values()))
                return m(r.astype(dtype))
        else:
            mapper = compose(m, slicerf)

        # convert our initial to a list
        return it.chain(mapper(initial),
                        it.chain.from_iterable(map(mapper, reader)))

    def _extend(self, rows):
        mode = 'ab' if PY2 else 'a'
        dialect = keyfilter(to_csv_kwargs.__contains__, self.dialect)

        f = self.open(self.path, mode)

        try:
            # we have data in the file, append a newline
            if os.path.getsize(self.path):
                f.write('\n')

            for df in map(partial(bz.into, pd.DataFrame),
                          partition_all(self.chunksize, iter(rows))):
                df.to_csv(f, index=False, header=None, **dialect)
        finally:
            try:
                f.close()
            except AttributeError:
                pass

    def remove(self):
        """Remove the persistent storage."""
        os.unlink(self.path)


@dispatch(CSV)
def drop(c):
    c.remove()


@resource.register('.*\.(csv|data|txt|dat)')
def resource_csv(uri, **kwargs):
    return CSV(uri, **kwargs)


@resource.register('.*\.(csv|data|txt|dat)\.gz')
def resource_csv_gz(uri, **kwargs):
    return CSV(uri, open=gzip.open, **kwargs)
