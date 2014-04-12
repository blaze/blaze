from __future__ import absolute_import, division, print_function

import csv
import itertools as it
import os

import datashape
from dynd import nd

from .. import py2help
from .data_descriptor import DDesc, Capabilities
from .dynd_data_descriptor import DyND_DDesc
from .as_py import ddesc_as_py
from .util import coerce
from ..utils import partition_all
from .. import py2help


def open_file(path, mode, has_header):
    """Return a file handler positionated at the first valid line."""
    csvfile = open(path, mode=mode)
    if has_header:
        csvfile.readline()
    return csvfile


def csv_descriptor_iterchunks(filename, mode, has_header, schema,
                              blen, dialect={}, start=None, stop=None):
    with open_file(filename, mode, has_header) as f:
        f = it.islice(csv.reader(f, **dialect), start, stop)
        for rows in partition_all(blen, f):
            # TODO: better way to define dshape?
            dshape = str(len(rows)) + ' * ' + schema
            yield DyND_DDesc(nd.array(rows, dtype=dshape))


def coerce_record_to_row(schema, rec):
    """

    >>> from datashape import dshape

    >>> schema = dshape('{x: int, y: int}')
    >>> coerce_record_to_row(schema, {'x': 1, 'y': 2})
    [1, 2]
    """
    return [rec[name] for name in schema[0].names]


class CSV_DDesc(DDesc):
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
    has_header : boolean
        Whether the CSV file has a header or not.  If not specified a value
        is guessed.
    """

    def __init__(self, path, mode='r', schema=None, dialect=None,
            has_header=None, **kwargs):
        if 'r' in mode and os.path.isfile(path) is not True:
            raise ValueError('CSV file "%s" does not exist' % path)
        self.path = path
        self.mode = mode
        csvfile = open(path, mode=self.mode)

        # Handle Schema
        if isinstance(schema, py2help._strtypes):
            schema = datashape.dshape(schema)
        if isinstance(schema, datashape.DataShape) and len(schema) == 1:
            schema = schema[0]
        self.schema = str(schema)

        # Handle Dialect
        if dialect is None and 'r' in mode:
            # Guess the dialect
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(csvfile.read(1024))
            except:
                pass
        if dialect is None:
            dialect = csv.get_dialect('excel')
        elif isinstance(dialect, py2help.basestring):
            dialect = csv.get_dialect(dialect)
        self.dialect = dict((key, getattr(dialect, key))
                            for key in dir(dialect) if not key.startswith('_'))

        # Update dialect with any keyword arguments passed in
        # E.g. allow user to override with delimiter=','
        for k, v in kwargs.items():
            if k in self.dialect:
                self.dialect[k] = v

        # Handle Header
        if has_header is None and mode != 'w':
            # Guess whether the file has a header or not
            sniffer = csv.Sniffer()
            csvfile.seek(0)
            sample = csvfile.read(1024)
            try:
                self.has_header = sniffer.has_header(sample)
            except:
                self.has_header = has_header

        else:
            self.has_header = has_header

        csvfile.close()

    @property
    def dshape(self):
        return datashape.DataShape(datashape.Var(), self.schema)

    @property
    def capabilities(self):
        """The capabilities for the csv data descriptor."""
        return Capabilities(
            # csv datadescriptor cannot be updated
            immutable = False,
            # csv datadescriptors are concrete
            deferred = False,
            # csv datadescriptor is persistent
            persistent = True,
            # csv datadescriptor can be appended efficiently
            appendable = True,
            remote = False,
            )

    def dynd_arr(self):
        # Positionate at the beginning of the file
        with open_file(self.path, self.mode, self.has_header) as csvfile:
            return nd.array(csv.reader(csvfile, **self.dialect), dtype=self.schema)


    def __array__(self):
        return nd.as_numpy(self.dynd_arr())

    def __len__(self):
        # We don't know how many rows we have
        return None

    def __getitem__(self, key):
        with open_file(self.path, self.mode, self.has_header) as csvfile:
            if isinstance(key, py2help._inttypes):
                start, stop, step = key, key + 1, 1
            elif isinstance(key, slice):
                start, stop, step = key.start, key.stop, key.step
            else:
                raise IndexError("key '%r' is not valid" % key)
            read_iter = csv.reader(it.islice(csvfile, start, stop, step),
                                   **self.dialect)
            res = nd.array(read_iter, dtype=self.schema)
        return DyND_DDesc(res)

    def __setitem__(self, key, value):
        # CSV files cannot be updated (at least, not efficiently)
        raise NotImplementedError

    def __iter__(self):
        with open(self.path, self.mode) as f:
            if self.has_header:
                next(f)  # burn header
            for row in csv.reader(f, **self.dialect):
                yield coerce(self.schema, row)

    def _extend(self, rows):
        rows = iter(rows)
        with open_file(self.path, self.mode, self.has_header) as f:
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

    def iterchunks(self, blen=100, start=None, stop=None):
        """Return chunks of size `blen` (in leading dimension).

        Parameters
        ----------
        blen : int
            The length, in rows, of the buffers that are returned.
        start : int
            Where the iterator starts.  The default is to start at the
            beginning.
        stop : int
            Where the iterator stops. The default is to stop at the end.

        Returns
        -------
        out : iterable
            This iterable returns buffers as DyND arrays,

        """
        # Return the iterable
        return csv_descriptor_iterchunks(
            self.path, self.mode, self.has_header,
            self.schema, blen, self.dialect, start, stop)

    def remove(self):
        """Remove the persistent storage."""
        os.unlink(self.path)
