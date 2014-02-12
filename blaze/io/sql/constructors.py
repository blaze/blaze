"""
SQL array constructors.
"""

from __future__ import absolute_import, division, print_function

from ... import Array
from .datadescriptor import SQLDataDescriptor

class SQLColumn(object):
    """Table and column name"""

    def __init__(self, table, colname):
        self.table = table
        self.colname = colname

    def __repr__(self):
        return "SQLColumn(%s)" % (self,)

    def __str__(self):
        return "%s.%s" % (self.table, self.colname)


def from_table(table, colname, dshape, conn):
    """
    Create a new blaze Array from a single column description.
    """
    col = SQLColumn(table, colname)
    return Array(SQLDataDescriptor(dshape, col, conn))

# TODO: Table