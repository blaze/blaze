========
Overview
========

Blaze Abstracts Computation and Storage
---------------------------------------

.. image:: svg/numpy_plus.png
    :align: center


The software ecosystem surrounding data analytics is rich with mature projects.
Blaze augments this ecosystem with a uniform interface.  Blaze orchestrates
computation and data access among these external projects.  It provides a
consistent backdrop to build standard interfaces usable by the current Python
community.


Datashape
---------

The type system in Blaze is called Datashape, and generalizes the
combination of shape and dtype in NumPy. The datashape of an array
consists of a number of dimensions, followed by an element type.

Data Descriptors
----------------

Data descriptors provide uniform data access.  They support
iteration, insertion, and fancy indexing over a variety of popular formats.
They provide seemless data migration and robust access for computation.
Data descriptors span from local memory to out of core storage to distributed
storage.

Expressions
-----------

Blaze expressions describe computational workflows symbolically. They allow
developers to architect and check their computations rapidly before applying
them to data.  These expressions can then be compiled down to a variety of
supported backends.

Backends
--------

Blaze backends include projects like streaming Python, Pandas, SQLAlchemy, and
Spark.  A Blaze expression can run equally well on any of these backends,
allowing developers to easily transition their computation as their needs
change.


Interfaces
----------

Blaze interfaces provide interactive Python objects focused on usability.
These high level ``Table`` and ``Array`` objects manage Blaze expressions and
computations in an interactive session similar to existing workflows with
Pandas DataFrames and NumPy NDArrays.
