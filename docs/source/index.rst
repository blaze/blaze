Blaze
=====

Blaze is the next generation of NumPy, Python’s extremely popular
array library. Blaze is designed to handle out-of-core computations
on large datasets that exceed the system memory capacity, as well as
on distributed and streaming data.

Blaze will allow analysts and scientists to productively write robust
and efficient code, without getting bogged down in the details of how
to distribute computation, or worse, how to transport and convert data
between databases, formats, proprietary data warehouses, and other
silos.

The core of Blaze consists of generic multi-dimensional Array and Table objects
===============================================================================
with an associated type system for expressing all kinds of data
types and layouts, especially semi-structured, sparse, and columnar
data.  Blaze’s generalized calculation engine can iterate over the
distributed array or table and dispatch to low-level kernels specialized
for the layout and type of the data.

Documentation Note
~~~~~~~~~~~~~~~~~~

As you read this documentation, please be aware that the project
is under development. The documentation and implementation of Blaze
have gone through many changes, and there are many places where they
have not been properly brought into sync. 

Index
~~~~~

.. toctree::
   :maxdepth: 2

   install
   overview
   quickstart
   catalog
   data_sources
   datashape
   datashape-binary
   data_descriptor
   blaze_function
   deferred
   dev_workflow

Original Index
~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   vision
   tutorials
   persistence
   memory
   module

API Reference
~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   toplevel
   types
   eclass
   typeinference
   layout
   sources
   graph
   ops
   table
   releases
   legal

Developer Guide
~~~~~~~~~~~~~~~

.. toctree::
   extending

Index
~~~~~

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
