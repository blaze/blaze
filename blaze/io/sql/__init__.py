from __future__ import absolute_import, division, print_function

try:
    #import pyodbc as db
    import sqlite3 as db
except ImportError:
    db = None

from .constructors import from_table
from .kernel import SQL
from . conn import connect
from . import sql_interp
from .datadescriptor import SQLDataDescriptor

# --- Initialize ---
from . import ufuncs