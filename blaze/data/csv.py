from __future__ import absolute_import, division, print_function

import csv
import itertools as it
import os

import datashape
from dynd import nd

from .core import DataDescriptor
from .utils import coerce_record_to_row
from ..utils import partition_all, nth
from .. import py2help

__all__ = ['CSV']


def has_header(sample):
    """

    >>> s = '''
    ... x,y
    ... 1,1
    ... 2,2'''
    >>> has_header(s)
    True
    """
    sniffer = csv.Sniffer()
    try:
        return sniffer.has_header(sample)
    except:
        return None


def discover_dialect(sample, dialect=None, **kwargs):
    """

    >>> s = '''
    ... 1,1
    ... 2,2'''
    >>> discover_dialect(s) # doctest: +SKIP
    {'escapechar': None,
     'skipinitialspace': False,
     'quoting': 0,
     'delimiter': ',',
     'lineterminator': '\r\n',
     'quotechar': '"',
     'doublequote': False}
    """
    if isinstance(dialect, py2help._strtypes):
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
    A Blaze data descriptor which exposes a CSV file.

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
    """
    immutable = False
    deferred = False
    persistent = True
    appendable = True
    remote = False

    def __init__(self, path, mode='r', schema=None, dialect=None,
            header=None, open=open, **kwargs):
        if 'r' in mode and os.path.isfile(path) is not True:
            raise ValueError('CSV file "%s" does not exist' % path)
        self.path = path
        self.mode = mode
        self.open = open

        if not schema:
            # TODO: Infer schema
            raise ValueError('No schema detected')
        self._schema = schema

        if os.path.exists(path) and mode != 'w':
            with self.open(path, 'r') as f:
                sample = f.read(1024)
        else:
            sample = ''
        dialect = discover_dialect(sample, dialect, **kwargs)
        assert dialect
        if header is None:
            header = has_header(sample)

        self.header = header
        self.dialect = dialect

    def _getitem(self, key):
        with self.open(self.path, self.mode) as f:
            if self.header:
                next(f)
            if isinstance(key, py2help._inttypes):
                line = nth(key, f)
                result = next(csv.reader([line], **self.dialect))
            elif isinstance(key, slice):
                start, stop, step = key.start, key.stop, key.step
                result = list(csv.reader(it.islice(f, start, stop, step),
                                         **self.dialect))
            else:
                raise IndexError("key '%r' is not valid" % key)
        return result

    def _iter(self):
        with self.open(self.path, self.mode) as f:
            if self.header:
                next(f)  # burn header
            for row in csv.reader(f, **self.dialect):
                yield row

    def _extend(self, rows):
        rows = iter(rows)
        with self.open(self.path, self.mode) as f:
            if self.header:
                next(f)
            row = next(rows)
            if isinstance(row, dict):
                schema = datashape.dshape(self.schema)
                row = coerce_record_to_row(schema, row)
                rows = (coerce_record_to_row(schema, row) for row in rows)

            # Write all rows to file
            f.seek(0, os.SEEK_END)  # go to the end of the file
            writer = csv.writer(f, **self.dialect)
            writer.writerow(row)
            writer.writerows(rows)

    def remove(self):
        """Remove the persistent storage."""
        os.unlink(self.path)
