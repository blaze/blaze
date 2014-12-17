==============================
Interacting with SQL Databases
==============================

How to
------

Typically one provides a SQL connection string to the ``Data`` constructor

.. code-block:: python

   >>> db = Data('postgresql:///user:pass@hostname')  # doctest: +SKIP

   or

   >>> t = Data('postgresql://user:pass@hostname::my-table-name')  # doctest: +SKIP

Alternatively users familiar with SQLAlchemy can pass any SQLAlchemy engine,
metadata, or Table objects to ``Data``.  This can be useful if you need to
specify more information that does not fit comfortably into a URI (like a
desired schema.)

   >>> import sqlalchemy
   >>> engine = sqlalchemy.create_engine('postgreqsql://hostname')

   >>> db = Data(engine)

How does it work?
-----------------

As you manipulate a Blaze expression Blaze in turn manipulates a SQLAlchemy
expression.  When you ask for a result SQLAlchemy generates the SQL appropriate
for your database and sends the query to the database to be run.


What databases does Blaze support?
----------------------------------

Blaze derives all SQL support from SQLAlchemy so really one should ask, *What
databases does SQLAlchemy support?*.  The answer is *quite a few* in the main
SQLAlchemy project and *most* when you include third party libraries.

However, URI support within Blaze is limited to a smaller set.  For exotic
databases you may have to create a ``sqlalchemy.engine`` explicitly as shown
above.


What operations work on SQL databases?
--------------------------------------

Most tabular operations, but not all.  SQLAlchemy translation is a high
priority. Failures include array operations like slicing and dot products don't
make sense in SQL.  Additionally some operations like datetime access are not
yet well supported through SQLAlchemy.  Finally some databases, like SQLite,
have limited support for common mathematical functions like ``sin``.
