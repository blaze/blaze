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

Symbol
------

A ``Symbol`` refers to a single collection of data.  It must be given a name
and a datashape.

.. code-block:: python

   >>> from blaze import *
   >>> accounts = Symbol('accounts', 'var * {id: int, name: string, balance: int}')


Projections, Selection, Arithmetic
----------------------------------

Many operations follow from standard Python syntax, familiar from systems like
NumPy and Pandas.

The following example defines a collection, ``accounts``, and then selects the
names of those accounts with negative balance.

.. code-block:: python

   >>> accounts = Symbol('accounts', 'var * {id: int, name: string, balance: int}')

   >>> deadbeats = accounts[accounts.balance < 0].name

Internally this doesn't do any actual work (we haven't specified a data
source.)  Instead it builds a symbolic representation of a comutation to
execute in the future.

.. code-block:: python

   >>> deadbeats
   accounts[accounts.balance < 0].name

   >>> deadbeats.dshape
   dshape("var * string")

Split-apply-combine, Reductions
-------------------------------

Blaze borrows the ``by`` operation from ``R`` and ``Julia``.  The ``by``
operation is a combined ``groupby`` and reduction, fulfilling
split-apply-combine workflows.

.. code-block:: python

   >>> by(accounts.name,                 # Splitting/grouping element
   ...    total=accounts.balance.sum())  # Apply and reduction
   by(accounts.name, total=sum(accounts.balance))


This operation groups the collection by name and then sums the balance of each
group.  It finds out how much all of the "Alice"s, "Bob"s, etc. of the world
have in total.

Note the reduction ``sum`` in the third apply argument.  Blaze supports the
standard reductions of numpy like ``sum``, ``min``, ``max`` and also the
reductions of Pandas like ``count`` and ``nunique``.


Join
----

Collections can be joined with the ``join`` operation, which allows for advanced
queries to span multiple collections.

.. code-block:: python

   >>> accounts = Symbol('accounts', 'var * {id: int, name: string, balance: int}')
   >>> cities = Symbol('cities', 'var * {name: string, city: string}')

   >>> join(accounts, cities, 'name')
   Join(lhs=accounts, rhs=cities, _on_left='name', _on_right='name', how='inner')

If given no inputs, ``join`` will join on all columns with shared names between
the two collections.

   >>> shared_names = join(accounts, cities)

Other
-----

Blaze supports a variety of other operations common to our supported backends.
See our API docs for more details.
