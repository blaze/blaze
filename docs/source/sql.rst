==============================
Interacting with SQL Databases
==============================

How to
------

Typically one provides a SQL connection string to the ``data`` constructor

.. code-block:: python

   >>> db = data('postgresql:///user:pass@hostname')  # doctest: +SKIP

   or

   >>> t = data('postgresql://user:pass@hostname::my-table-name')  # doctest: +SKIP

Alternatively users familiar with SQLAlchemy can pass any SQLAlchemy engine,
metadata, or Table objects to ``data``.  This can be useful if you need to
specify more information that does not fit comfortably into a URI (like a
desired schema.)

.. code-block:: python

   >>> import sqlalchemy  # doctest: +SKIP
   >>> engine = sqlalchemy.create_engine('postgreqsql://hostname')  # doctest: +SKIP

   >>> db = data(engine)  # doctest: +SKIP

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


How can I try this out?
-----------------------

The easiest way to play with SQL is to download a SQLite database.  We
recommend the `Lahman baseball statistics database`_.  After downloading one could connect blaze
to that database with the following code

.. code-block:: python

   >>> from blaze import data
   >>> db = data('sqlite:///Downloads/lahman2013.sqlite') # doctest: +SKIP
   >>> db.<tab>  # see available tables  # doctest: +SKIP
   db.AllstarFull          db.FieldingOF           db.Schools              db.fields
   db.Appearances          db.FieldingPost         db.SchoolsPlayers       db.isidentical
   db.AwardsManagers       db.HallOfFame           db.SeriesPost           db.like
   db.AwardsPlayers        db.Managers             db.Teams                db.map
   db.AwardsShareManagers  db.ManagersHalf         db.TeamsFranchises      db.relabel
   db.AwardsSharePlayers   db.Master               db.TeamsHalf            db.schema
   db.Batting              db.Pitching             db.apply                db.temp
   db.BattingPost          db.PitchingPost         db.data
   db.Fielding             db.Salaries             db.dshape
   >>> db.Teams.peek()  # view one particular database  # doctest: +SKIP
       yearID lgID teamID franchID divID  Rank   G  Ghome   W   L     ...       \
   0     1871   NA    BS1      BNA  None     3  31    NaN  20  10     ...
   1     1871   NA    CH1      CNA  None     2  28    NaN  19   9     ...
   2     1871   NA    CL1      CFC  None     8  29    NaN  10  19     ...
   3     1871   NA    FW1      KEK  None     7  19    NaN   7  12     ...

       DP    FP                     name                               park  \
   0  NaN  0.83     Boston Red Stockings                South End Grounds I
   1  NaN  0.82  Chicago White Stockings            Union Base-Ball Grounds
   2  NaN  0.81   Cleveland Forest Citys       National Association Grounds
   3  NaN  0.80     Fort Wayne Kekiongas                     Hamilton Field

       attendance  BPF  PPF  teamIDBR  teamIDlahman45  teamIDretro
   0          NaN  103   98       BOS             BS1          BS1
   1          NaN  104  102       CHI             CH1          CH1
   2          NaN   96  100       CLE             CL1          CL1
   3          NaN  101  107       KEK             FW1          FW1
   ...

One can then query and compute results as with a normal blaze workflow.


Connecting to a Schema Other than ``public`` with PostgreSQL
------------------------------------------------------------

To connect to a non-default schema, one may pass a ``sqlalchemy.MetaData``
object to ``data``. For example:


.. code-block:: python

   >>> from blaze import data
   >>> from sqlalchemy import MetaData
   >>> ds = data(MetaData('postgresql://localhost/test', schema='my_schema'))
   >>> ds.dshape  # doctest: +SKIP
   dshape("{table_a: var * {a: ?int32}, table_b: var * {b: ?int32}}")

.. _`Lahman baseball statistics database`: https://github.com/jknecht/baseball-archive-sqlite/raw/master/lahman2013.sqlite


Foreign Keys and Automatic Joins
--------------------------------

Often times one wants to access the columns of a table into which we have a foreign key.

For example, given a ``products`` table with this schema:

  .. code-block:: sql

     create table products (
         id integer primary key,
         name text
     )

and an ``orders`` table with this schema:

  .. code-block:: sql

     create table orders (
         id integer primary key,
         product_id integer references (id) products,
         quantity integer
     )

we want to get the name of the products in every order. In SQL, you would write the following join:

  .. code-block:: sql

     select
         o.id, p.name
     from
         orders o
             inner join
         products p
             on o.product_id = p.id


This is fairly straightforward. However, when you have more than two joins the SQL
gets unruly and hard to read. What we really want is a syntactically simply way to
follow the chain of foreign key relationships and be able to access columns in
foreign tables without having to write a lot of code. This is where blaze comes in.

Blaze can generate the above joins for you, so instead of writing a bunch of joins in
SQL you can simply access the columns of a foreign table as if they were columns on
the foreign key column.

The previous example in blaze looks like this:

  .. code-block:: python

     >>> from blaze import data, compute
     >>> d = data('postgresql://localhost/db')  # doctest: +SKIP
     >>> d.fields  # doctest: +SKIP
     ['products', 'orders']
     >>> expr = d.orders.product_id.name  # doctest: +SKIP
     >>> print(compute(expr))  # doctest: +SKIP
     SELECT orders.id, p.name
     FROM orders as o, products as p
     WHERE o.product_id = p.id


.. warning::

   The above feature is very experimental right now. We would
   appreciate bug reports and feedback on the API.
