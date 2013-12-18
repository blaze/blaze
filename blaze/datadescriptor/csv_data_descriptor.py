from __future__ import absolute_import

import csv
import itertools as it

from .data_descriptor import IDataDescriptor
from .. import datashape
from dynd import nd
from .dynd_data_descriptor import DyNDDataDescriptor


def csv_descriptor_iter(csvfile, schema):
    for row in csv.reader(csvfile):
        yield DyNDDataDescriptor(nd.array(row, dtype=schema))


def csv_descriptor_iterchunks(csvfile, schema, blen, start=None, stop=None):
    if blen == 1:
        # In this case we will return single rows, not a list of rows
        for nrow, row in enumerate(csv.reader(csvfile)):
            if start is not None and nrow < start: continue
            if stop is not None and nrow >= stop: return
            yield DyNDDataDescriptor(nd.array(row, dtype=schema))
    else:
        # The most general form
        rows = []
        for nrow, row in enumerate(csv.reader(csvfile)):
            if start is not None and nrow < start: continue
            if stop is not None and nrow >= stop:
                # Build the descriptor for the data we have and return
                yield DyNDDataDescriptor(nd.array(rows, dtype=schema))
                return
            rows.append(row)
            if nrow % blen == 0:
                yield DyNDDataDescriptor(nd.array(rows, dtype=schema))
                rows = []


class CSVDataDescriptor(IDataDescriptor):
    """
    A Blaze data descriptor which exposes a CSV file.

    Parameters
    ----------
    csvfile : file IO handle
        A file handler for the CSV file.
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

    def __init__(self, csvfile, schema=None, **kwargs):
        if not hasattr(csvfile, "__iter__"):
            raise TypeError('csvfile does not have an iter interface')
        self.csvfile = csvfile
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
            self.dialect = sniffer.sniff(csvfile.read(1024))
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
    def persistent(self):
        return True

    @property
    def is_concrete(self):
        """Returns True, CSV arrays are concrete."""
        return True

    @property
    def dshape(self):
        return datashape.dshape('Var, %s' % self.schema)

    @property
    def writable(self):
        return False

    @property
    def appendable(self):
        return True

    @property
    def immutable(self):
        return False

    def dynd_arr(self):
        # Positionate at the beginning of the file
        self.reset_file()
        return nd.array(csv.reader(self.csvfile), dtype=self.schema)

    def __array__(self):
        return nd.as_numpy(self.dynd_arr())

    def __len__(self):
        # We don't know how many rows we have
        return None

    def reset_file(self):
        """Positionated at the first valid line."""
        self.csvfile.seek(0)
        if self.has_header:
            self.csvfile.readline()
    
    def __getitem__(self, key):
        self.reset_file()
        if type(key) in (int, long):
            start, stop, step = key, key + 1, 1
        elif type(key) is slice:
            start, stop, step = key.start, key.stop, key.step
        else:
            raise IndexError("key '%r' is not valid" % key)
        read_iter = it.islice(csv.reader(self.csvfile), start, stop, step)
        res = nd.array(read_iter, dtype=self.schema)
        return DyNDDataDescriptor(res)

    def __setitem__(self, key, value):
        # CSV files cannot be updated (at least, not efficiently)
        raise NotImplementedError

    def __iter__(self):
        # Positionate at the beginning of the file
        self.reset_file()
        return csv_descriptor_iter(self.csvfile, self.schema)

    def append(self, row):
        """Append a row of values (in sequence form)."""
        self.csvfile.seek(0, 2)  # go to the end of the file
        values = nd.array(row, dtype=self.schema)  # validate row
        writer = csv.writer(self.csvfile, dialect=self.dialect)
        writer.writerow([unicode(v) for v in values])

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
        self.reset_file()
        return csv_descriptor_iterchunks(self.csvfile, self.schema,
                                         blen, start, stop)
