from __future__ import print_function, division, absolute_import

data = [
    (4,  "hello", 2.1),
    (8,  "world", 4.2),
    (16, "!",     8.4),
]


def create_sqlite_table():
    import sqlite3 as db

    conn = db.connect(":memory:")
    c = conn.cursor()
    c.execute('''create table testtable
    (i INTEGER, msg text, price real)''')
    c.executemany("""insert into testtable
                  values (?, ?, ?)""", data)
    conn.commit()
    c.close()

    return conn

#def create_sqlite_table():
#    import pyodbc as db
#    conn = db.connect("Driver=SQLite ODBC Driver "
#                      "NameDatabase=Database8;LongNames=0;Timeout=1000;"
#                      "NoTXN=0;SyncPragma=NORMAL;StepAPI=0;")
#    #conn = db.connect("Data Source=:memory:;Version=3;New=True;")
#    c = conn.cursor()
#    c.execute('''create table testtable
#    (i INTEGER, msg text, price real)''')
#    c.executemany("""insert into testtable
#                  values (?, ?, ?)""", data)
#    conn.commit()
#    c.close()
#
#    return conn
