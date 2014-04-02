"""
SQL array constructors.
"""

from __future__ import absolute_import, division, print_function

from ... import Array
from .datadescriptor import SQL_DDesc

from datashape import dshape, Record, DataShape, coretypes


class TableSelection(object):
    """
    Table and column name

    Attributes
    ==========

    table: str
        table name

    colname: str
        column name
    """

    def __init__(self, table_name, colname):
        self.table_name = table_name
        self.col_name = colname

    def __repr__(self):
        return "TableSelection(%s)" % (self,)

    def __str__(self):
        return "%s.%s" % (self.table_name, self.col_name)


def sql_table(table_name, colnames, measures, conn):
    """
    Create a new blaze Array from an SQL table description. This returns
    a Record array.

    Parameters
    ==========

    table_name: str
        table name

    colnames: [str]
        column names

    measures: [DataShape]
        measure (element type) for each column

    conn: pyodbc/whatever Connection
    """
    dtype = Record(list(zip(colnames, measures)))
    record_dshape = DataShape(coretypes.Var(), dtype)
    table = TableSelection(table_name, '*')
    return Array(SQL_DDesc(record_dshape, table, conn))


def sql_column(table_name, colname, dshape, conn):
    """
    Create a new blaze Array from a single column description.

    Parameters
    ==========

    table_name: str
        table name

    colname: str
        column

    dshape: DataShape
        type for the column. This should include the dimension, which may be
        a TypeVar

    conn: pyodbc/whatever Connection
    """
    col = TableSelection(table_name, colname)
    return Array(SQL_DDesc(dshape, col, conn))
