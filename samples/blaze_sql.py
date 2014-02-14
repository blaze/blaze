# -*- coding: utf-8 -*-

"""
This example shows some examples of how to access data in SQL databases using
blaze. It walks through how blaze syntax corresponds to SQL queries.

Select Queries
--------------


"""

from __future__ import absolute_import, division, print_function

import sqlite3 as db

from blaze.io.sql import connect, sql_column
from blaze import dshape

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
    c.executemany("""insert into testtable
                  values (?, ?, ?)""", data)
    conn.commit()
    c.close()

    return conn

conn = create_sqlite_table()
# Describe the columns. Note: typically you would describe column
# with variables for the column size, e.g. dshape('a, int32')

id = sql_column('MyTable', 'id', dshape('3, int32'), conn)
name_col = sql_column('MyTable', 'name', dshape('3, int32'), conn)
age_col = sql_column('MyTable', 'age', dshape('3, int32'), conn)

table = Table([id, name_col, age_col]) # TODO: Better interface


def select():
    """
    SELECT * FROM MyTable WHERE MyTable.id > 5
    """
    print(table[table.id > 5])


def select_ordered():
    """
    SELECT * FROM MyTable WHERE MyTable.id > 5 ORDER BY MyTable.age
    """
    print(index(table, table.id > 5, order=table.age))


def groupby():
    """
    SELECT *
    FROM MyTable
    WHERE MyTable.age > 10 AND MyTable.age < 20
    GROUP BY MyTable.age
    """
    teenagers = index(table, table.age > 10 & table.age < 20)
    print(groupby(teenagers, teenagers.age))


def aggregate():
    """
    SELECT AVG(age) FROM MyTable WHERE MyTable.id > 5
    """
    print(avg(age_col[id > 5]))