================
Data Descriptors
================


Data Descriptors provide uniform access to a variety of common data formats.
They provide standard iteration, insertion, and numpy-like fancy indexing over
on disk files in common formats like csv, json, and hdf5 in memory data
strutures like Python list-of-lists and DyND arrays as well as more
sophisticated data stores like SQL databases.  The data descriptor interface is
analogous to the Python buffer interface described in PEP 3118, but with some
more flexibility.

Over the course of this document we'll refer to the following simple csv file:

::

   # accounts.csv
   id, name, balance
   1, Alice, 100
   2, Bob, 200
   3, Charlie, 300
   4, Denis, 400
   5, Edith, 500

::

   >>> csv = CSV('accounts.csv')

Python-Style Iteration
======================

Data descriptors expose the ``__iter__`` method, which iterates over the
outermost dimension of the data.  This iterator yields vanilla Python objects
by default.

::

   >>> list(csv)
   [(1L, u'Alice', 100L),
    (2L, u'Bob', 200L),
    (3L, u'Charlie', 300L),
    (4L, u'Denis', 400L),
    (5L, u'Edith', 500L)]


Data descriptors also expose a ``chunks`` method, which also iterates over the
outermost dimension but instead of yielding single rows of Python objects
instead yields larger chunks of compactly stored data.  These chunks emerge as
DyND arrays which are more efficient for bulk processing and data transfer.
DyND arrays support the ``__array__`` interface and so can be easily converted
to NumPy arrays.

::

   >>> next(csv.chunks())
   nd.array([[1, "Alice", 100],
             [2, "Bob", 200],
             [3, "Charlie", 300],
             [4, "Denis", 400],
             [5, "Edith", 500]],
            type="5 * {id : int64, name : string, balance : int64}")

Insertion
=========

Analagously to ``__iter__`` and ``chunks`` the methods ``extend`` and
``extend_chunks`` allow for insertion of data into the data descriptor.  These
methods take iterators of Python objects and DyND arrays respectively.  The
data is coerced into whatever form is native for the storage medium e.g. text
for CSV or ``INSERT`` statements for SQL.


::

   >>> csv = CSV('accounts.csv', mode='a')
   >>> csv.extend([(6, 'Frank', 600),
   ...             (7, 'Georgina', 700)])


Migration
=========

The combination of iteration and insertion enables trivial data migration
between storage formats.

::

   >>> sql = SQL('postgres://user:password@hostname/', 'accounts')
   >>> sql.extend(iter(csv))  # Migrate csv file to Postgres database


Indexing
========

Data descriptors also support fancy indexing.  As with iteration this supports
either Python objects or DyND arrays with the ``.py[...]`` and ``.dynd[...]``
interfaces.

::

   >>> list(csv.py[::2, ['name', 'balance']])
   [(u'Alice', 100L),
    (u'Charlie', 300L),
    (u'Edith', 500L),
    (u'Georgina', 700L),
    (u'Georgina', 700L)]

   >>> csv.dynd[::10, ['column_1', 'column_3']]
   nd.array([["Alice", 100],
             ["Charlie", 300],
             ["Edith", 500],
             ["Georgina", 700]],
            type="var * {name : string, balance : int64}")

Performance of this approach varies depending on the underlying storage system.
For file-based storage systems like CSV and JSON we must seek through the file
to find the right line (see iopro_), but don't incur deserialization costs.
Some storage systems, like HDF5, support random access natively.


.. _iopro: http://docs.continuum.io/iopro/index.html
