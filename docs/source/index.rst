Blaze
=====

Blaze is the next generation of NumPy, Python’s extremely popular
array library. Blaze is designed to handle out-of-core computations
on large datasets that exceeds the system memory capacity, as well as
distributed and streaming data.

Blaze will allow analysts and scientists to productively write robust
and efficient code, without getting bogged down in the details of how
to distribute computation, or worse, how to transport and convert data
between databases, formats, proprietary data warehouses, and other
silos.

The core of Blaze is a generic N-dimensional array/table object with
a very general “data type” and “data shape” to a robust type
system for all kinds of data; especially semi-structured, sparse, and
columnar data. Blaze’s generalized calculation engine can iterate
over the distributed array or table and dispatch to low-level kernels,
selected via the dynamic data typing mechanism.

Contents
--------

.. toctree::
   :maxdepth: 2

   install
   overview
   quickstart
   desc
   blaze
   datashape
   memory
   persistence
   sources
   graph
   ops
   typechecker
   table
   execution
   releases
   legal

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

