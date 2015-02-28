###############################
Constraints and Column Metadata
###############################


Systems
*******

Various systems allow users to place constraints on the organization of their data. The most well-known and well-studied example of this is relational database management systems, or RDBMSs. Constraints allow one to do a variety of things, but one thing that stands out is their ability to enforce that certain properties and dependencies are always maintained. This document seeks to enumerate at high level the ways in which constraints are written down--and to some degree their function--across a few different systems.

We are going to compare SQL systems and a less well-known but very effective RDBMS: KDB+ (and its driver language ``q``).


Keys
****
Keys allow a database to identify each row in a table uniquely. There is only a single primary key in a table and it must be unique and contain no ``NULL`` values. On the other hand, a multitude of unique keys can exist in a table and can be defined on columns that allow ``NULL`` values.

We show how two systems write down keys.

In SQL we define our primary key when we define our table:

.. code-block:: sql

   create table t (
       id integer primary key,
       name varchar(128) not null,
       amount real not null
   );

   insert into t values ('Alice', 100.0), ('Bob', 200.0), ('Joe', -400.0);

Similarly in ``q``:

::

   q) t: ([id: 1 2 3] name: `Alice`Bob`Joe; amount: 100 200 -400)


Foreign Keys
============
Many (if not all) relational database systems provide a way to relate columns in possibly different tables to one another. When a column in one table refers to a column in another table we call this a *foreign key*.

Foreign keys are indicated by using a table's name (in this case ``t``) as the type of a field when constructing another table:

Here's how we write them down.

SQL:

.. code-block:: sql

   create table s (
       t_id integer not null,
       classes varchar(200) not null,
       number real,
       foreign key (t_id) references t
   );

   insert into s values (1, 'a', 1.0), (2, 'b', 2.0), (3, 'c', NULL), (1, 'd', NULL), (2, 'e', 5.0);

``q``:

::

   q) s: ([]; t_id: `t$1 2 3 1 2; classes: `a`b`c`d`e; number: 1.0 2.0 0n 0n 5.0)

In ``q`` this allows one to do things like an implicit left join, referencing any column in the table referred to by the foreign key:

::

   q) select t_id.name from s
   name
   ----
   b
   c
   a
   b
   c

With SQL, we must be more explicit:

.. code-block:: sql

   select t.name from s left join t on s.t_id = t.id;


Other Metadata
==============

Additionally, KDB+ exposes metadata regarding the structure of a particular column. In particular it allows one to attribute different properties to a column. Those properties are:

Sorted
------

This indicates that a column in sorted in ascending order. Lookups will be done with binary search rather than a linear scan of the column and operations that can take advantage of sortedness (e.g., ``min`` and ``max``) will be greatly sped up.

Unique
------

* ``q``: Uniqueness indicates that the values are in a column are distinct. Certain operations such as ``distinct`` and ``nunique`` are sped up many times.
* SQL: These are specified using the keyword ``unique``. The performance implications will be dependent on the RDBMS being used.

Grouped
-------

According to the ``q`` documentation the Grouped attributed roughly corresponds to a SQL index. It is implemented by using a hash table that maps unique values to a list of positions. This can speed up queries with a ``where`` clauses by allowing the query engine to immediately access the positions the query should run over.

Example:

.. code-block:: python

   >>> [0, 1, 2, 0, 1, 2, 1, 2, 0, 0, 1, 2, 0]  # lots of repetition, but not really any runs

In ``q``:

::

   q) `g#0 1 2 0 1 2 1 2 0 0 1 2 0


SQL


.. code-block:: sql

   create index my_index on t (column_name);

Parted
------

This attribute indicates that there are runs of the same value in our data. In this case ``q`` will store a hash table mapping unique values to the start position of each run. It likely stores the length of the run as well.

Example:

.. code-block:: python

   >>> [1, 1, -1, -1, -1, 2, 2, 2, 3]  # lots of repetition with runs

Implementation in the Blaze Ecosystem
=====================================

Regardless of how these properties change computation, ``blaze`` needs to be able to tell other systems about metadata where possible. This is one of the main benefits to using a relational database, namely, the ability to relate information.

Primary and Foreign Keys
------------------------

It is important to note that we want to keep foreign key metadata separate from type information. Adding foreign key metadata to a blaze expression will thus require an extra step.

   .. code-block:: python

      >>> t = symbol('t', dshape=..., meta={'keys': {'primary': ['id']}})
      >>> s = symbol('s', dshape=..., meta={'keys': {'foreign': {'t_id': t.id}}})

This approach has the benefit of being low tech. We're passing around core Python data structures (``list`` and ``dict``) that fully describe primary and foreign key relationships. The ability to describe metadata isn't hampered by having to learn yet another API.

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

      >>> select t.amount from s left join t on s.t_id = t.id;

or in ``q``:

   .. code-block:: sql

      >>> select t_id.amount from s

In the case of an interactive expression, when we construct objects we need to have tables with foreign keys keep a reference to the tables from which keys originate.

   .. code-block:: python

      >>> db = Data('sqlite:///path/to/db')  # contains s and t tables
      >>> compute(db.s.t_id.name)  # generates SQL similar to the above

Sorted
------
TODO

Unique and Grouped Attributes
-----------------------------

Categorical types can cover the cases of unique and grouped. We implement this in the ``datashape`` library. Categorical types have two relevant properties:

1. A list of unique sorted categories.
2. A list of the indices of the categories from item 1 as unsigned integers of the minimum type necessary to support the number of unique categories.

Parted
------
TODO
