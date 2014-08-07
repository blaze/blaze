===========
Expressions
===========

Blaze expressions describe computational workflows symbolically. They allow
developers to architect and check their computations rapidly before applying
them to data.  These expressions can then be compiled down to a variety of
supported backends.

Tables
======

Table expressions track operations found in relational algebra or your standard
Pandas/R DataFrame object.  Operations include projecting columns, filtering, mapping and basic mathematics, reductions, split-apply-combine (groupby) operations, and joining.  This compact set of operations can express a surprisingly large set of common computations.  They are widely supported.

TableSymbol
-----------

A ``TableSymbol`` refers to a single table in storage.  It must be given a name
and a schema.

.. code-block:: python

   >>> accounts = TableSymbol('accounts', '{id: int, name: string, balance: int}')


Projections, Selection, Arithmetic
----------------------------------

Intuitive operations follow from intuitive Python syntax, learned from systems
like NumPy and Pandas.  The following defines a table, ``accounts``, and then
selects the names of those accounts with negative balance.

.. code-block:: python

   >>> accounts = TableSymbol('accounts', '{id: int, name: string, balance: int}')

   >>> deadbeats = accounts[accounts['balance'] < 0]['name']

Internally this doesn't do any actual work (we haven't specified a data source
or a computational engine).  Instead it builds up a symbolic representation of
a comutation to do in the future.

.. code-block:: python

   >>> deadbeats
   accounts[accounts['balance'] < 0]['name']

   >>> deadbeats.schema
   dshape("{name: string}")

Split-apply-combine, Reductions
-------------------------------

Blaze borrows the ``by`` operation from ``R`` and ``Julia``.  The ``by``
operation is a combined ``groupby`` and reduction, fulfilling
split-apply-combine workflows.

.. code-block:: python

   >>> by(accounts,                     # Table
   ...    accounts['name'],             # Splitting/grouping element
   ...    accounts['balance'].sum())    # Apply and reduction

This operation groups the table by name and then sums the balance of each
group.  It finds out how much all of the "Alice"s, "Bob"s, etc. of the world
have in total.

Note the reduction ``sum`` in the third apply argument.  Blaze supports the
standard reductions of numpy like ``sum``, ``min``, ``max`` and also the
reductions of Pandas like ``count`` and ``nunique``.


Join
----

Tables can be joined with the ``join`` operation, which allows for advanced
queries to span multiple tables.

.. code-block:: python

   >>> accounts = TableSymbol('accounts', '{name: string, balance: int}')
   >>> cities = TableSymbol('cities', '{name: string, city: string}')

   >>> join(accounts, cities, 'name')

If given no inputs, ``join`` will join on all columns with shared names between
the two tables.

   >>> join(accounts, cities)


Other
-----

Blaze supports a variety of other operations common to our supported backends.
See our API docs for more details.
