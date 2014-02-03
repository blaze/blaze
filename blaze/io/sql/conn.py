"""
SQL connection and naming interface.

TODO: instantiate this stuff from the catalog?
"""

from __future__ import absolute_import, division, print_function
from . import db

#------------------------------------------------------------------------
# Connect
#------------------------------------------------------------------------

def connect(odbc_conn_str):
    """Connect to a SQL database using an ODBC connection string"""
    return db.connect(odbc_conn_str)