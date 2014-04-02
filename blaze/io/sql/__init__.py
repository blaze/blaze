from __future__ import absolute_import, division, print_function

try:
    #import pyodbc as db
    import sqlite3 as db
except ImportError:
    db = None

from .constructors import sql_table, sql_column
from .kernel import SQL
from . conn import connect
from .datadescriptor import SQL_DDesc

# --- Initialize ---
from . import ops
