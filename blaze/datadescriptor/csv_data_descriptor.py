from __future__ import absolute_import, division, print_function

import csv
import itertools as it
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


def csv_descriptor_iter(filename, mode, has_header, schema):
    with open_file(filename, mode, has_header) as csvfile:
        for row in csv.reader(csvfile):
            yield DyND_DDesc(nd.array(row, dtype=schema))


def csv_descriptor_iterchunks(filename, mode, has_header, schema,
                              blen, start=None, stop=None):
    rows = []
    with open_file(filename, mode, has_header) as csvfile:
        for nrow, row in enumerate(csv.reader(csvfile)):
            if start is not None and nrow < start:
                continue
            if stop is not None and nrow >= stop:
                if rows != []:
                    # Build the descriptor for the data we have and return
                    yield DyND_DDesc(nd.array(rows, dtype=schema))
                return
            rows.append(row)
            if nrow % blen == 0:
                print("rows:", rows, schema)
                yield DyND_DDesc(nd.array(rows, dtype=schema))
                rows = []


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

    def __init__(self, path, mode='r', **kwargs):
        if os.path.isfile(path) is not True:
            raise ValueError('CSV file "%s" does not exist' % path)
        self.path = path
        self.mode = mode
        csvfile = open(path, mode=self.mode)
        schema = kwargs.get("schema", None)
        dialect = kwargs.get("dialect", None)
        has_header = kwargs.get("has_header", None)
        if type(schema) in py2help._strtypes:
            schema = datashape.dshape(schema)
        if isinstance(schema, datashape.DataShape) and len(schema) == 1:
            schema = schema[0]
        if not isinstance(schema, datashape.Record):
            raise TypeError(
                'schema cannot be converted into a blaze record dshape')
        self.schema = str(schema)

        if dialect is None:
            # Guess the dialect
            sniffer = csv.Sniffer()
            try:
                self.dialect = sniffer.sniff(csvfile.read(1024))
            except:
                # Cannot guess dialect.  Assume Excel.
                self.dialect = csv.get_dialect('excel')
            csvfile.seek(0)
        else:
            if isinstance(dialect, csv.Dialect):
                self.dialect = dialect
            else:
                self.dialect = csv.get_dialect(dialect)
        if has_header is None:
            # Guess whether the file has a header or not
            sniffer = csv.Sniffer()
            self.has_header = sniffer.has_header(csvfile.read(1024))
            csvfile.seek(0)
        else:
            self.has_header = has_header

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
            return nd.array(csv.reader(csvfile), dtype=self.schema)

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
            read_iter = it.islice(csv.reader(csvfile), start, stop, step)
            res = nd.array(read_iter, dtype=self.schema)
        return DyND_DDesc(res)

    def __setitem__(self, key, value):
        # CSV files cannot be updated (at least, not efficiently)
        raise NotImplementedError

    def __iter__(self):
        return csv_descriptor_iter(
            self.path, self.mode, self.has_header, self.schema)

    def append(self, row):
        """Append a row of values (in sequence form)."""
        values = nd.array(row, dtype=self.schema)  # validate row
        with open_file(self.path, self.mode, self.has_header) as csvfile:
            csvfile.seek(0, os.SEEK_END)  # go to the end of the file
            delimiter = self.dialect.delimiter
            csvfile.write(delimiter.join(py2help.unicode(v) for v in row)+'\n')

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
            self.schema, blen, start, stop)
