"""
This example shows some examples of how to access data in SQL databases using
blaze. It walks through how blaze syntax corresponds to SQL queries.

Select Queries
--------------


"""

from __future__ import absolute_import, division, print_function

import sqlite3 as db

from datashape import dshape
from blaze.io.sql import sql_table


def create_sqlite_table():
    data = [
        (4,  "Gilbrecht", 17),
        (8,  "Bertrand", 48),
        (16, "Janssen", 32),
    ]

    conn = db.connect(":memory:")
    c = conn.cursor()
    c.execute('''create table MyTable
    (id INTEGER, name TEXT, age INTEGER)''')
    c.executemany("""insert into MyTable
                  values (?, ?, ?)""", data)
    conn.commit()
    c.close()

    return conn

conn = create_sqlite_table()

# Describe the columns. Note: typically you would describe column
# with variables for the column size, e.g. dshape('a, int32')
table = sql_table('MyTable',
                  ['id', 'name', 'age'],
                  [dshape('int32'), dshape('string'), dshape('float64')],
                  conn)
