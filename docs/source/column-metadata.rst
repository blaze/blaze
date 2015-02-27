***************
Column Metadata
***************

This document is heavily inspired by the KDB+ DBMS and its driver language ``q``.

In ``q``, tables are constructed like this:

   .. code-block:: sh

      q) t: ([] name: `Alice`Bob`Joe; amount: 100 200 -400)

Primary Keys
============
To use a primary key, we construct the table by filling in the square brackets.

   .. code-block:: sh

      q) t: ([id: 1 2 3] name: `Alice`Bob`Joe; amount: 100 200 -400)

Foreign Keys
============
Many (if not all) database systems provide a way to relate columns in possibly different tables to each other. When a column in one table refers to a column in another table we call this a *foreign key*.

Foreign keys are indicated by using a table's name (in this case ``t``) as the type of a field when constructing another table:

   .. code-block:: sh

      q) s: ([]; t_id: `t$1 2 3 1 2; classes: `a`b`c`d`e; number: 1.0 2.0 0n 0n 5.0)


This allows one to do things like an implicit join, referencing any column in the table referred to by the foreign key:

   .. code-block:: sh

      q) select t_id.name from s
      name
      ----
      b
      c
      a
      b
      c


Other Metadata
==============

Additionally, KDB+ exposes metadata regarding the structure of a particular column. In particular it allows one to attribute different properties to a column. Those properties are:

Sorted
------

This indicates that a column in sorted in ascending order. Lookups will be done with binary search rather than a linear scan of the column and operations that can take advantage of sortedness (e.g., ``min`` and ``max``) will be greatly sped up.

Unique
------

Uniqueness indicates that the values are in a column are distinct. Certain operations such as ``distinct`` and ``nunique`` are sped up many times.

Grouped
-------

According to the ``q`` documentation the Grouped attributed roughly corresponds to a SQL index. It is implemented by using a hash table that maps unique values to a list of positions. This can speed up queries with a ``where`` clauses by allowing the query engine to immediately access the positions the query should run over.

Example:

.. code-block:: python

   >>> [0, 1, 2, 0, 1, 2, 1, 2, 0, 0, 1, 2, 0]  # lots of repetition, but not really any runs

Parted
------

This attribute indicates that there are runs of the same value in our data. In this case ``q`` will store a hash table mapping unique values to the start position of each run. It likely stores the length of the run as well.

Example:

.. code-block:: python

   >>> [1, 1, -1, -1, -1, 2, 2, 2, 3]  # lots of repetition with runs

Implementation
==============

Regardless of how these properties change computation, ``blaze`` needs to be able to tell other systems about metadata where possible. This is one of the main benefits to using a relational database, namely, the ability to relate information.

Sorted
------

TODO

Unique and Grouped Attributes
-----------------------------

Categorical Types
^^^^^^^^^^^^^^^^^
Categorical types can cover the cases of unique and grouped. We implement this in the ``datashape`` library. It has two relevant properties:

1. A list of unique sorted categories.
2. A list of the indices of the categories from item 1 as unsigned integers of the minimum type necessary to support the number of unique categories.

Primary and Foreign Keys
------------------------

It is important to note that we want to keep foreign key metadata separate from type information. Adding foreign key metadata to a blaze expression will thus require an extra step.

   .. code-block:: python

      >>> t = symbol('t', """var * {
      ...     id: int64,
      ...     name: categorical[type=string, categories=["Alice", "Bob", "Joel"]],
      ...     amount: float64
      ... }""")
      >>> t.primary_key = t.id
      >>> s = symbol('s', "var * {t_id: int64, height: float64}")
      >>> s.relations = {s.t_id: t.id}


Alternatively we could allow ``primary_key`` and ``relations`` keyword arguments:

   .. code-block:: python

      >>> t = symbol('t', """var * {
      ...     id: int64,
      ...     name: categorical[type=string, categories=["Alice", "Bob", "Joel"]],
      ...     amount: float64
      ... }""", primary_key='id')
      >>> s = symbol('s', "var * {t_id: int64, height: float64}", relations={'t_id': t.id})


A more general approach might be to pass around a dictionary of arbitrary metadata. This has the benefit of not tying the notion of relations and keys to blaze's ``Symbol`` objects, which are meant to be able to describe many different kinds of data other than tables.


   .. code-block:: python

      >>> t = symbol('t', dshape=..., meta={'keys': {'primary': ['id']}})
      >>> s = symbol('s', dshape=..., meta={'keys': {'foreign': {'t_id': t.id}}})

This approach also has the benefit of being low tech. We're passing around core Python data structures (``list`` and ``dict``) that fully describe primary and foreign key relationships.


Validation can also take place at construction time. A few things can be validated:

* The ``keys`` dictionary key is only valid on ``Record`` dshapes
* The primary key is actually in the datashape
* The primary key is a valid primary key type (this would likely be limited to integers at first)
* Foreign keys all reference primary keys


In any case we want to be able to access implicit join functionality:

   .. code-block:: python

      >>> compute(s.t_id.amount, {s: <sdata>, t: <tdata>})

The previous call to ``compute`` would return the equivalent of

   .. code-block:: sql

      >>> select amount from s left join t on s.t_id = t.id;

or in ``q``:

   .. code-block:: sql

      >>> select t_id.amount from s

In the case of an interactive expression, when we construct objects we need to have tables with foreign keys keep a reference to the tables from which keys originate.

   .. code-block:: python

      >>> t = Data('sqlite:///path/to/db::t')
      >>> s = Data('sqlite:///path/to/db::s')  # s has a foreign key into t

      >>> compute(s.t_id.name)  # generates SQL similar to the above

In the case of ``into`` the syntax is slightly different

   .. code-block:: python

      >>> t = into('sqlite:///path/to/db::t', pd.DataFrame(...),
      ...          primary_key='id')
      >>> s = into('sqlite:///path/to/db::s', pd.DataFrame(...),
      ...          foreign_keys={'t_id': 't.id'})
