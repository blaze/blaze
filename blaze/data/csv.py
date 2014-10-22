from __future__ import absolute_import, division, print_function

import itertools as it
import os
import gzip
import bz2
from functools import partial
from contextlib import contextmanager

from ..dispatch import dispatch
from cytoolz import partition_all, merge, keyfilter, pluck
from toolz import concat, get, pipe, identity, take
from toolz.curried import map, get
from dynd import nd

import pandas as pd

from datashape.discovery import discover, null, unpack
from datashape import (dshape, Record, Option, Fixed, Unit, Tuple, string,
                       DataShape)
import datashape as ds
from datashape.predicates import isdimension

import blaze as bz
from .pandas_dtype import dshape_to_pandas
from .core import DataDescriptor
from ..resource import resource
from ..utils import nth, nth_list, keywords
from .. import compatibility
from ..compatibility import SEEK_END, builtins, _strtypes, _inttypes
from ..compatibility import zip, PY2
from .utils import ordered_index, listpack, coerce

import csv

__all__ = ['CSV', 'drop']


numtypes = frozenset(ds.integral.types) | frozenset(ds.floating.types)
na_values = frozenset(pd.io.parsers._NA_VALUES)


read_csv_kwargs = set(keywords(pd.read_csv))
assert read_csv_kwargs

def clean_dialect(dialect):
    """ Make a csv dialect apprpriate for pandas.read_csv """
    dialect = keyfilter(read_csv_kwargs.__contains__,
            dialect)
    # handle windows
    if dialect['lineterminator'] == '\r\n':
        dialect['lineterminator'] = None

    return dialect

to_csv_kwargs = set(keywords(pd.core.format.CSVFormatter.__init__))
assert to_csv_kwargs

DEFAULT_ENCODING = 'utf-8'

def has_header(sample, encoding=DEFAULT_ENCODING):
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
    try:
        dialect = csv.get_dialect(dialect)
    except csv.Error:
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel

    assert dialect is not None

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

    # line_terminator is for to_csv
    dialect['lineterminator'] = dialect['line_terminator'] = \
            dialect.get('line_terminator', dialect.get('lineterminator', os.linesep))
    return dialect


@contextmanager
def csvopen(csv, **kwargs):
    try:
        f = csv.open(csv.path, encoding=csv.encoding, **kwargs)
    except (TypeError, ValueError):  # TypeError for py2 ValueError for py3
        f = csv.open(csv.path, **kwargs)

    yield f

    try:
        f.close()
    except AttributeError:
        pass


def get_sample(csv, size=16384):
    path = csv.path

    if os.path.exists(path) and csv.mode != 'w':
        with csvopen(csv, mode='rt') as f:
            return f.read(size)
    return ''


def ext(path):
    _, e = os.path.splitext(path)
    return e.lstrip('.')


def safely_option(ds):
    """ Wrap certain types in an option type

    >>> safely_option('int32')
    ?int32
    >>> safely_option('?int32')
    ?int32
    >>> safely_option('float64')
    ctype("float64")
    """
    if isinstance(ds, _strtypes):
        ds = dshape(ds)
    if isinstance(ds, DataShape) and len(ds) == 1:
        ds = ds[0]
    if isinstance(ds, Unit) and 'int' in str(ds) or 'date' in str(ds):
        return Option(ds)
    return ds


def discover_csv(path, encoding=DEFAULT_ENCODING, nrows_discovery=50,
        header=None, dialect=None, types=None, columns=None,
        typehints=None):
    """ Discover datashape of CSV file """
    df = pd.read_csv(path,
            dtype='O',
            encoding=encoding,
            chunksize=nrows_discovery,
            compression={'gz': 'gzip',
                         'bz2': 'bz2'}.get(ext(path)),
            header=0 if header else None,
            **clean_dialect(dialect)).get_chunk()
    if not types:
        L = (df.fillna('')
                .to_records(index=False)
                .tolist())
        rowtype = discover(L).subshape[0]
        if isinstance(rowtype[0], Tuple):
            types = rowtype[0].dshapes
            types = [unpack(t) for t in types]
            types = [string if t == null else t for t in types]
            types = [safely_option(t) for t in types]
        elif (isinstance(rowtype[0], Fixed) and
                isinstance(rowtype[1], Unit)):
            types = int(rowtype[0]) * [rowtype[1]]
        else:
            raise ValueError("Could not discover schema from data.\n"
                    "Please specify schema.")
    if not columns:
        if header:
            columns = list(df.columns)
        else:
            columns = ['_%d' % i for i in range(len(types))]
    if typehints:
        types = [typehints.get(c, t) for c, t in zip(columns, types)]

    return dshape(Record(list(zip(columns, types))))



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
            encoding=DEFAULT_ENCODING, **kwargs):
        if 'r' in mode and not os.path.isfile(path):
            raise ValueError('CSV file "%s" does not exist' % path)

        if schema is None and 'w' in mode:
            raise ValueError('Please specify schema for writable CSV file')

        self.path = path
        self.mode = mode
        self.open = {'gz': gzip.open, 'bz2': bz2.BZ2File}.get(ext(path), open)
        self._abspath = os.path.abspath(path)
        self.chunksize = chunksize
        self.encoding = encoding

        sample = get_sample(self)
        self.dialect = dialect = discover_dialect(sample, dialect, **kwargs)

        if header is None:
            header = has_header(sample, encoding=encoding)
        elif isinstance(header, int):
            header = True
        self.header = header

        if not schema and 'w' not in mode:
            schema = discover_csv(path, encoding=encoding, dialect=dialect,
                    header=self.header, typehints=typehints,
                    types=types, columns=columns,
                    nrows_discovery=nrows_discovery)
        self._schema = schema
        self.header = header

        if 'w' not in mode:
            try:
                nd.array(list(take(10, self._iter(chunksize=10))),
                         dtype=str(schema))
            except TypeError as e:
                raise ValueError("Automatic datashape discovery failed\n"
                        "Discovered the following datashape: %s\n"
                        "But DyND generated the following error: %s\n"
                        "Consider providing type hints using "
                        "typehints={'column-name': 'type'}\n"
                        "like typehints={'start-time': 'string'}"
                        % (schema, e.args[0]))


    def get_py(self, key):
        return self._get_py(ordered_index(key, self.dshape))

    def _get_py(self, key):
        if isinstance(key, tuple):
            assert len(key) == 2
            rows, cols = key
            usecols = cols
            ds = self.dshape.subshape[rows, cols]
            usecols = None if isinstance(usecols, slice) else listpack(usecols)
        else:
            rows = key
            ds = self.dshape.subshape[rows]
            usecols = None

        if isinstance(ds, DataShape) and isdimension(ds[0]):
            ds = ds.subshape[0]

        seq = self._iter(usecols=usecols)
        if isinstance(key, tuple) and isinstance(cols, _strtypes + _inttypes):
            seq = pluck(0, seq)
        seq = coerce(ds, seq)

        if isinstance(rows, compatibility._inttypes):
            line = nth(rows, seq)
            try:
                return next(line).item()
            except TypeError:
                try:
                    return line.item()
                except AttributeError:
                    return line
        elif isinstance(rows, list):
            return nth_list(rows, seq)
        elif isinstance(rows, slice):
            return it.islice(seq, rows.start, rows.stop, rows.step)
        else:
            raise IndexError("key %r is not valid" % rows)

    def pandas_read_csv(self, usecols=None, **kwargs):
        """ Use pandas.read_csv with the right keyword arguments

        In particular we know what dtypes should be, which columns are dates,
        etc...
        """
        dtypes, dates = dshape_to_pandas(self.schema)

        if usecols:
            if builtins.all(isinstance(c, int) for c in usecols):
                usecols = get(usecols, self.columns)
            dates = [name for name in dates if name in usecols]

        header = kwargs.pop('header', self.header)
        header = 0 if self.header else None

        result = pd.read_csv(self.path,
                             names=kwargs.pop('names', self.columns),
                             usecols=usecols,
                             compression={'gz': 'gzip',
                                          'bz2': 'bz2'}.get(ext(self.path)),
                             dtype=kwargs.pop('dtype', dtypes),
                             parse_dates=kwargs.pop('parse_dates', dates),
                             encoding=kwargs.pop('encoding', self.encoding),
                             header=header,
                             **merge(kwargs, clean_dialect(self.dialect)))

        reorder = get(list(usecols)) if usecols and len(usecols) > 1 else identity

        if isinstance(result, (pd.Series, pd.DataFrame)):
            return reorder(result)
        else:
            return map(reorder, result)

    def _iter(self, usecols=None, chunksize=None):
        from blaze.api.into import into
        chunksize = chunksize or self.chunksize
        dfs = self.pandas_read_csv(usecols=usecols,
                                   chunksize=chunksize,
                                   dtype='O',
                                   parse_dates=[])
        return pipe(dfs, map(partial(pd.DataFrame.fillna, value='')),
                         map(partial(into, list)),
                         concat)

    def last_char(self):
        r"""Get the last character of the file.

        Warning
        -------
        * This method should not be used when the file :attr:`~blaze.CSV.path`
          is already open.

        Notes
        -----
        Blaze's CSV data descriptor :meth:`~blaze.CSV.extend` method differs
        from both pandas' (to_csv) and python's (csv.writer.writerow(s)) CSV
        writing tools. Both of these libraries assume a newline at the end of
        the file when appending and are not robust to data that may or may not
        have a newline at the end of the file.

        In our case we want users to be able to make multiple calls to extend
        without having to worry about this annoying detail, like this:

        ::

            a.extend(np.ndarray)
            a.extend(tables.Table)
            a.extend(pd.DataFrame)


        Another way to put it is calling :meth:`~blaze.CSV.extend` on this

            a,b\n1,2\n

        and this

            a,b\n1,2

        should do the same thing, thus the need to know the last character in
        the file.
        """
        if not os.path.exists(self.path) or not os.path.getsize(self.path):
            return os.linesep

        offset = len(os.linesep)

        # read in binary mode to allow negative seek indices, but return an
        # encoded string
        with csvopen(self, mode='rb') as f:
            f.seek(-offset, SEEK_END)
            return f.read(offset).decode(self.encoding)

    def _extend(self, rows):
        mode = 'ab' if PY2 else 'a'
        newline = dict() if PY2 else dict(newline='')
        dialect = keyfilter(to_csv_kwargs.__contains__, self.dialect)
        should_write_newline = self.last_char() != os.linesep
        with csvopen(self, mode=mode, **newline) as f:
            # we have data in the file, append a newline
            if should_write_newline:
                f.write(os.linesep)

            for df in map(partial(bz.into, pd.DataFrame),
                          partition_all(self.chunksize, iter(rows))):
                df.to_csv(f, index=False, header=None, encoding=self.encoding,
                          **dialect)

    def remove(self):
        """Remove the persistent storage."""
        os.unlink(self.path)


@dispatch(CSV)
def drop(c):
    c.remove()


@resource.register('.+\.(\wsv|data|txt|dat)')
def resource_csv(uri, **kwargs):
    return CSV(uri, **kwargs)


@resource.register('.+\.(\wsv|data|txt|dat)\.gz')
def resource_csv_gz(uri, **kwargs):
    return CSV(uri, open=gzip.open, **kwargs)
