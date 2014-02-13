"""
SQL data descriptor using pyodbc.
"""

from __future__ import absolute_import, division, print_function
from itertools import chain

from ... import Array, array
from ...datadescriptor import DyNDDataDescriptor
from ..storage import Storage
from ...datadescriptor import IDataDescriptor, Capabilities
from .query import execute, dynd_chunk_iterator

from datashape import DataShape

from dynd import nd, ndt

class SQLDataDescriptor(IDataDescriptor):
    """
    SQL data descriptor. This describes a column of some SQL table.
    """

    def __init__(self, dshape, col, conn):
        """
        Parameters
        ----------

        col: SQLColumn
            Holds an SQL table name from which we can select data. This may also
            be some other valid query on which we can do further selection etc.
        """
        assert dshape
        assert col
        assert conn
        self._dshape = dshape
        self.col = col
        self.conn = conn

        # TODO: Validate query as a suitable expression to select from

    @property
    def dshape(self):
        return self._dshape

    @property
    def capabilities(self):
        """The capabilities for the SQL data descriptor."""
        return Capabilities(
            immutable = True,
            deferred = False,
            persistent = True,
            appendable = False,
            remote=True,
            )

    def describe_col(self):
        query_result = execute(self.conn, self.dshape,
                               "select %s from %s", (self.col.colname,
                                                     self.col.table))
        return SQLResultDataDescriptor(query_result)

    def __iter__(self):
        return iter(self.describe_col())

    def __getitem__(self, item):
        raise NotImplementedError

    def dynd_arr(self):
        return self.describe_col().dynd_arr()

    def __repr__(self):
        return "SQLDataDescriptor(%s)" % (self.col,)

    def __str__(self):
        return str()

    _printer = __str__


class SQLResultDataDescriptor(IDataDescriptor):
    """
    SQL result data descriptor. This describes an query result and pulls it
    in lazily.
    """

    _dynd_result = None

    def __init__(self, query_result):
        assert query_result
        self._dshape = query_result.dshape
        self.query_result = _ResultIterable(query_result)

    @property
    def dshape(self):
        return self._dshape

    @property
    def capabilities(self):
        """The capabilities for the SQL result data descriptor."""
        return Capabilities(
            immutable = True,
            deferred = False,
            persistent = True,
            appendable = False,
            remote=True,
            )

    def __iter__(self):
        return iter((x for chunk in self.query_result for x in chunk))

    def __getitem__(self, item):
        raise NotImplementedError

    def dynd_arr(self):
        # TODO: This should really use blz
        if self._dynd_result is not None:
            return self._dynd_result

        # Allocate empty dynd array
        length = sum(len(chunk) for chunk in self.query_result)
        ds = DataShape(length, self.dshape.measure)
        result = nd.empty(str(ds))

        # Fill dynd array with chunks
        offset = 0
        for chunk in self.query_result:
            result[offset:offset + len(chunk)] = chunk
            offset += len(chunk)

        self._dynd_result = result
        return result

    def __repr__(self):
        return "SQLResultDataDescriptor(%s)" % (self.col,)

    def __str__(self):
        return str(Array(DyNDDataDescriptor(self.dynd_arr())))

    _printer = __str__


class _ResultIterable(object):
    """
    Pull query results from cursor into dynd. Can be iterated over as many
    times as necessary (iterable).
    """

    def __init__(self, query_result):
        self.query_result = _ResultIterator(query_result)

    def __iter__(self):
        return chain(self.query_result.chunks, self.query_result)


class _ResultIterator(object):
    """
    Pull query results from cursor into dynd. Can be iterated over once
    (iterator), after which all chunks are loaded in `self.chunks`.
    """

    def __init__(self, query_result):
        self.query_result = dynd_chunk_iterator(query_result)

        # Accumulated dynd chunks
        self.chunks = []

    def __iter__(self):
        return self

    def next(self):
        next_chunk = next(self.query_result)
        self.chunks.append(next_chunk)
        return next_chunk