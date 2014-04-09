from __future__ import absolute_import, division, print_function

import csv
import itertools as it
import toolz
import os

import datashape
from dynd import nd

from .. import py2help
from .data_descriptor import DDesc, Capabilities
from .dynd_data_descriptor import DyND_DDesc


def open_file(path, mode, has_header):
    """Return a file handler positionated at the first valid line."""
    csvfile = open(path, mode=mode)
    if has_header:
        csvfile.readline()
    return csvfile


def csv_descriptor_iter(filename, mode, has_header, schema, dialect={}):
    with open_file(filename, mode, has_header) as csvfile:
        for row in csv.reader(csvfile, **dialect):
            yield DyND_DDesc(nd.array(row, dtype=schema))


def csv_descriptor_iterchunks(filename, mode, has_header, schema,
                              blen, dialect={}, start=None, stop=None):
    with open_file(filename, mode, has_header) as f:
        f = it.islice(csv.reader(f, **dialect), start, stop)
        for rows in toolz.partition_all(blen, f):
            yield DyND_DDesc(nd.array(rows, dtype=schema))


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
        if mode == 'r' and os.path.isfile(path) is not True:
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
        if dialect is None and mode !='r':
            # Guess the dialect
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(csvfile.read(1024))
            except:
                pass
        if dialect is None:
            dialect = csv.get_dialect('excel')
        elif isinstance(dialect, (str, unicode)):
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
            self.has_header = sniffer.has_header(sample)
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
        return csv_descriptor_iter(
            self.path, self.mode, self.has_header, self.schema, self.dialect)

    def append(self, row):
        """Append a row of values (in sequence form)."""
        values = nd.array(row, dtype=self.schema)  # validate row
        with open_file(self.path, self.mode, self.has_header) as f:
            f.seek(0, os.SEEK_END)  # go to the end of the file
            writer = csv.writer(f, **self.dialect)
            writer.writerow(row)

    def extend(self, rows):
        """ Extend data with many rows

        See Also:
            append
        """
        rows = iter(rows)
        with open_file(self.path, self.mode, self.has_header) as f:
            # Validate first row
            row = next(rows)
            nd.array(row, dtype=self.schema)

            # Write all rows to file
            f.seek(0, os.SEEK_END)  # go to the end of the file
            writer = csv.writer(f, **self.dialect)
            writer.writerow(row)
            writer.writerows(rows)

    def iterchunks(self, blen=None, start=None, stop=None):
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
