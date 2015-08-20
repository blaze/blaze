========
Backends
========

Blaze backends include projects like streaming Python, Pandas, SQLAlchemy,
MongoDB, PyTables, and Spark.  Most Blaze expressions can run well on any of
these backends, allowing developers to easily transition their computation to
changing performance needs.

.. raw:: html

   <iframe style='width:105%; height:625px; border:0;' src='_static/html/capabilities.html'/>

Existing Backends
=================

Streaming Python
----------------

via `toolz <http://toolz.readthedocs.org/en/latest/>`_ and `cytoolz <https://github.com/pytoolz/cytoolz/>`_

Blaze can operate on core Python data structures like lists, tuples, ints and
strings.  This can be useful both in small cases like rapid prototyping or unit
testing but also in large cases where streaming computation is desired.

The performance of Python data structures like ``dict`` make Python a
surprisingly powerful platform for data-structure bound computations commonly
found in split-apply-combine and join operations.  Additionally, Python's
support for lazy iterators (i.e. generators) means that it can easily support
*streaming* computations that pull data in from disk, taking up relatively
little memory.

`Pandas <http://pandas.pydata.org>`_
------------------------------------

Pandas DataFrames are the gold standard for in-memory data analytics.  They are
fast, intuitive, and come with a wealth of additional features like plotting,
and data I/O.

`SQLAlchemy <http://www.sqlalchemy.org>`_
-----------------------------------------

Blaze levarages the SQLAlchemy project, which provides a uniform interface over
the varied landscape of SQL systems.  Blaze manipulates SQLAlchemy expressions
which are then compiled down to SQL query strings of the appropriate backend.

The prevalance of SQL among data technologies makes this backend particularly
useful.  Databases like Impala and Hive have SQLAlchemy dialects, enabling
easy Blaze interoperation.

`MongoDB <http://www.mongodb.org/>`_
-------------------------------------

Blaze drives MongoDB through the `pymongo
<http://api.mongodb.org/python/current/api/pymongo/index.html>`_ interface and
is able to use many of the built in operations such as aggregration and group
by.

`PyTables <http://www.pytables.org>`_
-------------------------------------

PyTables provides compressed Table objects backed by the popular HDF5 library.
Blaze can compute simple expressions using PyTables, such as elementwise
operations and row-wise selections.

`Spark <https://spark.apache.org/>`_
------------------------------------

Spark provides resilient distributed in-memory computing and easy access to
HDFS storage.  Blaze drives Spark through the `PySpark
<https://spark.apache.org/docs/0.9.0/python-programming-guide.html>`_
interface.

Benefits of Backend Agnostic Computation
========================================

For maximum performance and expressivity it is best to use the backends
directly.  Blaze is here when absolute customization is not required.

Familiarity
-----------

Users within the numeric Python ecosystem may be familiar with the NumPy and
Pandas interfaces but relatively unfamiliar with SQL or the functional idioms
behind Spark or Streaming Python.  In this case Blaze provides a familiar
interface which can drive common computations in these more exotic backends.

Prototyping and Testing
-----------------------

Blaze allows you to prototype and test your computation on a small dataset
using Python or Pandas and then scale that computation up to larger
computational systems with confidence that nothing will break.

A changing hardware landscape drives a changing software landscape.  Analytic
code written for systems today may not be relevant for systems five years from
now.  Symbolic systems like Blaze provide some stability on top of this
rapidly changing ecosystem.

Static Analysis
---------------

*Not yet implemented*

Blaze is able to inspect and optimize your computation before it is run.
Common optimizations include loop fusion, rearranging joins and projections to
minimize data flow, etc..
