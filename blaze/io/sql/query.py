"""
SQL query execution.
"""

from __future__ import absolute_import, division, print_function

from . import db

from datashape import DataShape, Record
from dynd import nd, ndt


def execute(conn, dshape, query, params):
    """
    Execute a query on the given connection and return a Result that
    can be iterated over or consumed in DyNd chunks.
    """
    cursor = conn.cursor()
    cursor.execute(query, params)
    return Result(cursor, dshape)


class Result(object):
    """
    Result from executing a query
    """

    def __init__(self, cursor, dshape):
        self.cursor = cursor
        self.dshape = dshape

    # def __iter__(self):
    #     return iter_result(self.cursor, self.dshape)


def iter_result(result, dshape):
    if not isinstance(dshape.measure, Record):
        return iter(row[0] for row in result)
    return iter(result)


def dynd_chunk_iterator(result, chunk_size=1024):
    """
    Turn a query Result into a bunch of DyND arrays
    """
    cursor = result.cursor

    chunk_size = max(cursor.arraysize, chunk_size)
    while True:
        try:
            results = cursor.fetchmany(chunk_size)
        except db.Error:
            break

        if not results:
            break

        dshape = DataShape(len(results), result.dshape.measure)
        chunk = nd.empty(str(dshape))
        chunk[:] = list(iter_result(results, dshape))
        yield chunk
