from __future__ import absolute_import
import operator
import contextlib
import ctypes
import csv
import itertools as it
import os

from .data_descriptor import IDataDescriptor, Capabilities
from .. import datashape
from dynd import nd, ndt
from .dynd_data_descriptor import DyNDDataDescriptor


def open_file(filename, has_header, mode='r'):
    """Return a file handler positionated at the first valid line."""
    csvfile = file(filename, mode=mode)
    if has_header:
        csvfile.readline()
    return csvfile


def csv_descriptor_iter(filename, has_header, schema):
    with open_file(filename, has_header) as csvfile:
        for row in csv.reader(csvfile):
            yield DyNDDataDescriptor(nd.array(row, dtype=schema))


def csv_descriptor_iterchunks(filename, has_header, schema,
                              blen, start=None, stop=None):
    rows = []
    with open_file(filename, has_header) as csvfile:
        for nrow, row in enumerate(csv.reader(csvfile)):
            if start is not None and nrow < start:
                continue
            if stop is not None and nrow >= stop:
                if rows != []:
                    # Build the descriptor for the data we have and return
                    yield DyNDDataDescriptor(nd.array(rows, dtype=schema))
                return
            rows.append(row)
            if nrow % blen == 0:
                print("rows:", rows, schema)
                yield DyNDDataDescriptor(nd.array(rows, dtype=schema))
                rows = []


class CSVDataDescriptor(IDataDescriptor):
    """
    A Blaze data descriptor which exposes a CSV file.

    Parameters
    ----------
    filename : string
        A path string for the CSV file.
    schema : string or blaze.datashape
        A blaze datashape (or its string representation) of the schema
        in the CSV file.
    dialect : string or csv.Dialect instance
        The dialect as understood by the `csv` module in Python standard
        library.  If not specified, a value is guessed.
    has_header : boolean
        Whether the CSV file has a header or not.  If not specified a value
        is guessed.
    """

    def __init__(self, filename, **kwargs):
        if os.path.isfile(filename) is not True:
            raise ValueError('CSV file "%s" does not exist' % filename)
        self.filename = filename
        csvfile = file(filename)
        schema = kwargs.get("schema", None)
        dialect = kwargs.get("dialect", None)
        has_header = kwargs.get("has_header", None)
        if type(schema) in (str, unicode):
            schema = datashape.dshape(schema)
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
            )

    def dynd_arr(self):
        # Positionate at the beginning of the file
        with open_file(self.filename, self.has_header) as csvfile:
            return nd.array(csv.reader(csvfile), dtype=self.schema)

    def __array__(self):
        return nd.as_numpy(self.dynd_arr())

    def __len__(self):
        # We don't know how many rows we have
        return None

    def __getitem__(self, key):
        with open_file(self.filename, self.has_header) as csvfile:
            if type(key) in (int, long):
                start, stop, step = key, key + 1, 1
            elif type(key) is slice:
                start, stop, step = key.start, key.stop, key.step
            else:
                raise IndexError("key '%r' is not valid" % key)
            read_iter = it.islice(csv.reader(csvfile), start, stop, step)
            res = nd.array(read_iter, dtype=self.schema)
        return DyNDDataDescriptor(res)

    def __setitem__(self, key, value):
        # CSV files cannot be updated (at least, not efficiently)
        raise NotImplementedError

    def __iter__(self):
        return csv_descriptor_iter(self.filename, self.has_header, self.schema)

    def append(self, row):
        """Append a row of values (in sequence form)."""
        with open_file(self.filename, self.has_header, mode='a') as csvfile:
            csvfile.seek(0, 2)  # go to the end of the file
            values = nd.array(row, dtype=self.schema)  # validate row
            delimiter = self.dialect.delimiter
            terminator = self.dialect.lineterminator
            csvfile.write(delimiter.join(unicode(v) for v in row)+terminator)

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
        return csv_descriptor_iterchunks(self.filename, self.has_header,
                                         self.schema, blen, start, stop)
