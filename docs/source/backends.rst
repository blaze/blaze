========
Backends
========

Blaze backends include projects like streaming Python, Pandas, SQLAlchemy, and
Spark.  A Blaze expression can run equally well on any of these backends,
allowing developers to easily transition their computation to changing
performance needs.

Existing Backends
=================

Streaming Python
----------------

Blaze can operate on core Python data structures like lists, tuples, ints and
strings.  This can be useful both in small cases like rapid prototyping or unit
testing but also in large cases where streaming computation is desired.

The preformant Python data structures like ``dict`` make Python a surprisingly
performant platform for data-structure bound computations commonly found in
split-apply-combine and join operations.  Additionally, Python's support for
lazy iterators (i.e. generators) means that it can easily support *streaming*
computations that pull data in from disk, taking up relatively little memory.

Pandas
------

Pandas DataFrames are the gold standard for in-memory data analytics.  They are
fast, intuitive, and come with a wealth of additional features like plotting,
and data I/O.

SQLAlchemy
----------

Blaze levarages the SQLAlchemy project, which provides a uniform interface over
the varied landscape of SQL systems.  Blaze manipulates SQLAlchemy expressions
which are then compiled down to SQL query strings of the appropriate backend.

Spark
-----

Spark provides resilient distributed in-memory computing and easy access to
HDFS storage.  Blaze drives Spark through the PySpark interface.


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
