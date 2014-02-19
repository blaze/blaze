"""
SQL array constructors.
"""

from __future__ import absolute_import, division, print_function

from ... import Array
from .datadescriptor import SQLDataDescriptor

from datashape import dshape, Record, DataShape

class TableSelection(object):
    """Table and column name"""

    def __init__(self, table, colname):
        self.table = table
        self.colname = colname

    def __repr__(self):
        return "TableSelection(%s)" % (self,)

    def __str__(self):
        return "%s.%s" % (self.table, self.colname)


def sql_table(table, colnames, measures, conn):
    """
    Create a new blaze Array from an SQL table description. This returns
    a Record array.
    """
    measure = Record(list(zip(colnames, measures)))
    record_dshape = DataShape(dshape('a'), measure)
    table = TableSelection(table, '*')
    return Array(SQLDataDescriptor(record_dshape, table, conn))


def sql_column(table, colname, dshape, conn):
    """
    Create a new blaze Array from a single column description.
    """
    col = TableSelection(table, colname)
    return Array(SQLDataDescriptor(dshape, col, conn))
