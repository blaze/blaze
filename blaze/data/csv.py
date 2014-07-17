from __future__ import absolute_import, division, print_function

import sys
if sys.version_info[0] == 2:
    import unicodecsv as csv
else:
    import csv
import itertools as it
import os
from operator import itemgetter
from collections import Iterator

import datashape
from datashape.discovery import discover, null, string, unpack
from datashape import dshape, Record, Option, Fixed, CType, Tuple, string
from dynd import nd

from .core import DataDescriptor
from .utils import coerce_record_to_row
from ..utils import nth, nth_list, get
from .. import compatibility
from ..compatibility import map

__all__ = ['CSV']


def has_header(sample):
    """ Sample text has a header """
    sniffer = csv.Sniffer()
    try:
        return sniffer.has_header(sample)
    except:
        return None


def discover_dialect(sample, dialect=None, **kwargs):
    """ Discover CSV dialect from string sample

    Returns dict
    """
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
    immutable = False
    deferred = False
    persistent = True
    appendable = True
    remote = False

    def __init__(self, path, mode='rt',
            schema=None, columns=None, types=None, typehints=None,
            dialect=None, header=None, open=open, nrows_discovery=50,
            **kwargs):
        if 'r' in mode and os.path.isfile(path) is not True:
            raise ValueError('CSV file "%s" does not exist' % path)
        if not schema and 'w' in mode:
            raise ValueError('Please specify schema for writable CSV file')
        self.path = path
        self.mode = mode
        self.open = open

        if os.path.exists(path) and mode != 'w':
            f = self.open(path)
            sample = f.read(16384)
            try:
                f.close()
            except AttributeError:
                pass
        else:
            sample = ''

        # Pandas uses sep instead of delimiter.
        # Lets support that too
        if 'sep' in kwargs:
            kwargs['delimiter'] = kwargs['sep']

        dialect = discover_dialect(sample, dialect, **kwargs)
        assert dialect
        if header is None:
            header = has_header(sample)

        if not schema and 'w' not in mode:
            if not types:
                with open(self.path) as f:
                    data = list(it.islice(csv.reader(f, **dialect), 1, nrows_discovery))
                    types = discover(data)
                    rowtype = types.subshape[0]
                    if isinstance(rowtype[0], Tuple):
                        types = types.subshape[0][0].dshapes
                        types = [unpack(t) for t in types]
                        types = [string if t == null else t
                                        for t in types]
                        types = [t if isinstance(t, Option) or t==string else Option(t)
                                      for t in types]
                    elif (isinstance(rowtype[0], Fixed) and
                          isinstance(rowtype[1], CType)):
                        types = int(rowtype[0]) * [rowtype[1]]
                    else:
                       ValueError("Could not discover schema from data.\n"
                                  "Please specify schema.")
            if not columns:
                if header:
                    with open(self.path) as f:
                        columns = next(csv.reader([next(f)], **dialect))
                else:
                    columns = ['_%d' % i for i in range(len(types))]
            if typehints:
                types = [typehints.get(c, t) for c, t in zip(columns, types)]

            schema = dshape(Record(list(zip(columns, types))))

        self._schema = schema

        self.header = header
        self.dialect = dialect

    def _get_py(self, key):
        if isinstance(key, tuple):
            assert len(key) == 2
            result = self._get_py(key[0])

            if isinstance(key[1], list):
                getter = itemgetter(*key[1])
            else:
                getter = itemgetter(key[1])

            if isinstance(key[0], (list, slice)):
                return map(getter, result)
            else:
                return getter(result)

        f = self.open(self.path)
        if self.header:
            next(f)
        if isinstance(key, compatibility._inttypes):
            line = nth(key, f)
            result = next(csv.reader([line], **self.dialect))
        elif isinstance(key, list):
            lines = nth_list(key, f)
            result = csv.reader(lines, **self.dialect)
        elif isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            result = csv.reader(it.islice(f, start, stop, step),
                                **self.dialect)
        else:
            raise IndexError("key '%r' is not valid" % key)
        try:
            if not isinstance(result, Iterator):
                f.close()
        except AttributeError:
            pass
        return result

    def _iter(self):
        f = self.open(self.path)
        if self.header:
            next(f)  # burn header
        for row in csv.reader(f, **self.dialect):
            yield row

        try:
            f.close()
        except AttributeError:
            pass

    def _extend(self, rows):
        rows = iter(rows)
        if sys.version_info[0] == 3:
            f = self.open(self.path, 'a', newline='')
        elif sys.version_info[0] == 2:
            f = self.open(self.path, 'ab')

        try:
            row = next(rows)
        except StopIteration:
            return
        if isinstance(row, dict):
            schema = dshape(self.schema)
            row = coerce_record_to_row(schema, row)
            rows = (coerce_record_to_row(schema, row) for row in rows)

        # Write all rows to file
        writer = csv.writer(f, **self.dialect)
        writer.writerow(row)
        writer.writerows(rows)

        try:
            f.close()
        except AttributeError:
            pass

    def remove(self):
        """Remove the persistent storage."""
        os.unlink(self.path)
