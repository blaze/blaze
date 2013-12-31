"""
SciDB connection and naming interface.

TODO: instantiate this stuff from the catalog?
"""

from __future__ import absolute_import, division, print_function

#------------------------------------------------------------------------
# Connect
#------------------------------------------------------------------------

class SciDBConn(object):
    """
    Refer to an individual SciDB array.
    """

    def __init__(self, conn):
        self.conn = conn

    def query(self, query, persist=False):
        return self.conn.execute_query(query, persist=persist)

    def wrap(self, arrname):
        return self.conn.wrap_array(arrname)


def connect(uri):
    """Connect to a SciDB database"""
    from scidbpy import interface
    return SciDBConn(interface.SciDBShimInterface(uri))
