"""
SQL array constructors.
"""

from __future__ import absolute_import, division, print_function

from ... import Array
from .datadescriptor import SQLDataDescriptor

def from_table(table, dshape, conn):
    """
    Create a new blaze Array from a single column description.
    """
    return Array(SQLDataDescriptor(dshape, table, conn))

# TODO: Table