"""
SQL data descriptor using pyodbc.
"""

from __future__ import absolute_import, division, print_function

from itertools import chain

from datashape import DataShape, Record
from dynd import nd

from ... import Array
from ...datadescriptor import DyND_DDesc
from ...datadescriptor import DDesc, Capabilities
from .query import execute, dynd_chunk_iterator


class SQL_DDesc(DDesc):
    """
    SQL data descriptor. This describes a column of some SQL table.
    """

    def __init__(self, dshape, col, conn):
        """
        Parameters
        ----------

        col: TableSelection
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
                               "select %s from %s" % (self.col.col_name,
                                                      self.col.table_name), [])
        return SQLResult_DDesc(query_result)

    def __iter__(self):
        return iter(self.describe_col())

    def __getitem__(self, item):
        """
        Support my_sql_blaze_array['sql_column']
        """
        from .constructors import sql_table, sql_column

        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], str):
            if item[0] != slice(None):
                raise NotImplementedError("Currently only allowing slicing of whole sql array.")
            table = self.col
            colname = item[1]

            assert table.col_name == '*'
            dshape = column_dshape(self.dshape, colname)

            # Create blaze array for remote column
            arr = sql_column(table.table_name, colname, dshape, self.conn)

            # Array.__getitem__ will expect back a data descriptor!
            return arr.ddesc

        raise NotImplementedError

    def dynd_arr(self):
        return self.describe_col().dynd_arr()

    def __repr__(self):
        return "SQL_DDesc(%s)" % (self.col,)

    def __str__(self):
        return "<sql col %s with shape %s>" % (self.col, self.dshape)

    _printer = __str__
    _printer_repr = __repr__


class SQLResult_DDesc(DDesc):
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
        return (x for chunk in self.query_result for x in chunk)

    def __getitem__(self, item):
        """
        Support my_sql_blaze_array['sql_column']
        """
        # TODO: Lazy column description
        # return self.dynd_arr()[item]

        if isinstance(item, str):
            # Pull in data to determine length
            # TODO: this is bad
            return DyND_DDesc(getattr(self.dynd_arr(), item))

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
        return "SQLResult_DDesc()"

    def __str__(self):
        return str(Array(DyND_DDesc(self.dynd_arr())))

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

    __next__ = next


def column_dshape(dshape, colname):
    """
    Given a record dshape, project a column out
    """
    rec = dshape.measure

    if not isinstance(rec, Record):
        raise TypeError("Can only select fields from record type")
    if colname not in rec.fields:
        raise ValueError("No such field %r" % (colname,))

    measure = rec.fields[colname]
    params = list(dshape.shape) + [measure]
    dshape = DataShape(*params)

    return dshape
