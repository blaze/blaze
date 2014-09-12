====
Data
====

Blaze Data Descriptors provide uniform access to a variety of common data
formats.  They provide standard iteration, insertion, and numpy-like fancy
indexing over on-disk files in common formats like csv, json, and hdf5 in
memory data strutures like core Python data structures and DyND arrays as well
as more sophisticated data stores like SQL databases.  The data descriptor
interface is analogous to the Python buffer interface described in PEP 3118,
but with some more flexibility.

Over the course of this document we'll refer to the following simple
``accounts.csv`` file:

::

   id, name, balance
   1, Alice, 100
   2, Bob, 200
   3, Charlie, 300
   4, Denis, 400
   5, Edith, 500

.. code-block:: python

   >>> from blaze import *
   >>> csv = CSV('examples/data/accounts.csv')

Interface
=========

Iteration
---------

Data descriptors expose the ``__iter__`` method, which iterates over the
outermost dimension of the data.  This iterator yields vanilla Python objects
by default.

.. code-block:: python

   >>> list(csv) #doctest: +SKIP
   [(1, 'Alice', 100), (2, 'Bob', 200), (3, 'Charlie', 300), (4, 'Denis', 400), (5, 'Edith', 500)]


Data descriptors also expose a ``chunks`` method, which also iterates over the
outermost dimension but instead of yielding single rows of Python objects
instead yields larger chunks of compactly stored data.  These chunks emerge as
DyND arrays which are more efficient for bulk processing and data transfer.
DyND arrays support the ``__array__`` interface and so can be easily converted
to NumPy arrays.

.. code-block:: python

   >>> next(csv.chunks())
   nd.array([  [1, "Alice", 100],     [2, "Bob", 200], [3, "Charlie", 300],
               [4, "Denis", 400],   [5, "Edith", 500]],
            type="5 * {id : ?int64, name : string, balance : ?int64}")

Insertion
---------

Analagously to ``__iter__`` and ``chunks`` the methods ``extend`` and
``extend_chunks`` allow for insertion of data into the data descriptor.  These
methods take iterators of Python objects and DyND arrays respectively.  The
data is coerced into whatever form is native for the storage medium e.g. text
for CSV or ``INSERT`` statements for SQL.


.. code-block:: python

   >>> csv = CSV('examples/data/accounts.csv', mode='a')    # doctest: +SKIP
   >>> csv.extend([(6, 'Frank', 600),                       # doctest: +SKIP
   ...             (7, 'Georgina', 700)])


Migration
---------

The combination of uniform iteration and insertion enables trivial data
migration between storage systems.

.. code-block:: python

   >>> sql = SQL('sqlite:///:memory:', 'accounts', schema=csv.schema)
   >>> sql.extend(iter(csv))  # Migrate csv file to SQLite database


Indexing
--------

Data descriptors also support fancy indexing.  As with iteration this supports
either Python objects or DyND arrays with the ``.py[...]`` and ``.dynd[...]``
interfaces.

.. code-block:: python

   >>> list(csv[::2, ['name', 'balance']])                  # doctest: +SKIP
   [('Alice', 100), ('Charlie', 300), ('Edith', 500), ('Georgina', 700)]

   >>> csv.dynd[2::, ['name', 'balance']]                   # doctest: +SKIP
   nd.array([ ["Charlie", 300],    ["Denis", 400],    ["Edith", 500],
                ["Frank", 600], ["Georgina", 700]],
            type="var * {name : string, balance : ?int64}")

Performance of this approach varies depending on the underlying storage system.
For file-based storage systems like CSV and JSON we must seek through the file
to find the right line (see iopro_), but don't incur deserialization costs.
Some storage systems, like HDF5, support random access natively.


Current State
=============


The ``blaze.data`` module robustly parses csv, json, hdf5 files and interacts
with SQL databases.

CSV/JSON
--------

For text-based formats (csv, json) it depends on standard Python modules
like ``csv`` to tokenize strings and the fast library DyND to serialize and
deserialize data elements.  This separation enables a *serialize what you need*
approach ideal for subsampling datasets.

.. code-block:: python

   >>> csv = CSV('examples/data/accounts.csv')
   >>> selection = csv[::2, 'name']  # Fast, deserializes a small fraction of dataset

HDF5
----

HDF5 support comes via h5py_, which loads data in through ``numpy`` arrays
and offers various forms of compression for binary data.

.. code-block:: python

   >>> hdf5 = HDF5('examples/data/accounts.h5', 'accounts')

Directories
-----------

Directories of files are supported with meta descriptors ``Stack`` and
``Concat`` which allow you to treat directories of files as a single, indexable
data source.

.. code-block:: python

   >>> from glob import glob
   >>> filenames = glob('examples/data/accounts*.csv')
   >>> csvs = [CSV(filename) for filename in filenames]

   >>> stack = Stack(csvs)
   >>> stack_slice = stack[:, ::2, 'name']

   >>> cat = Concat(csvs)
   >>> combined = cat[::2, 'name']

SQL
---

Robust SQL interaction is provided by SQLAlchemy_ which maps an abstract
expression system onto a variety of SQL backends including Postgres, MySQL,
SQLite, etc...

.. code-block:: python

   >>> sql = SQL('sqlite:///:memory:', 'accounts', schema='{id:int}')

Specifying Datashape
--------------------

Ideally Blaze is able to infer the schema/datashape of your dataset.  Systems
like SQL carry enough meta-data to ensure that this is possible.  Other systems
like CSV depend on heuristics.  These heurstics can fail or even err.  In that
case you may be prompted to provide more information

.. code-block:: python

   csv = CSV(filename)
   TypeError: Could not determine schema

   # Full schema specification as a datashape string
   csv = CSV(filename, schema='{id: int, name: string, amount: float32}')

   # Just specify the column names, please discover types
   csv = CSV(filename, columns=['id', 'name', 'amount'])

   # Provide corrections where needed
   csv = CSV(filename, columns=['id', 'name', 'amount'],
             typehints={'amount': 'float64'})



.. _iopro: http://docs.continuum.io/iopro/index.html
.. _h5py: http://docs.h5py.org/en/latest/
.. _SQLAlchemy: http://www.sqlalchemy.org/
