"""
SQL query execution.
"""

from __future__ import absolute_import, division, print_function
from datashape import DataShape
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

    def __iter__(self):
        return iter(self.cursor)


def dynd_chunk_iterator(result, chunk_size=1024):
    """
    Turn a query Result into a bunch of DyND arrays
    """
    cursor = result.curshor

    chunk_size = max(cursor.arraysize, chunk_size)
    dshape = DataShape(chunk_size, result.dshape.measure)
    while True:
        chunk = nd.empty(str(dshape))
        try:
            results = cursor.fetchmany(chunk_size)
        except db.Error:
            break

        chunk[:] = results
        yield chunk
