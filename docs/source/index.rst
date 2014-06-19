Blaze
=====

Blaze improves access to computational resources.  It does this by providing a
uniform and familiar interface to other computational projects.  Blaze
extends the familiar interfaces of NumPy and Pandas to external projects,
enabling easy access to out-of-core, streaming, and distributed computation.

Blaze allows analysts and scientists to productively write robust and efficient
code without getting bogged down in the details of how to distribute
computation, or convert data between databases, formats, proprietary data
warehouses, and other silos.


Core
----

The core of Blaze consists of

*   A type system to express data types and layouts
*   A symbolic expression system
*   A set of interfaces to common data formats
*   A set of interfaces to computational engines

This architecture allows a single Blaze code to run against several
computational backends.  Blaze depends on and exposes the hard work of
countless other projects.


Under Development
-----------------

Please be aware that Blaze is under active development. The project has gone
through many changes and this documentation has not been kept uniformly in
sync.

Index
~~~~~

.. toctree::
   :maxdepth: 2

   install
   overview
   quickstart
   data
   expressions
   backends
   datashape
   dev_workflow


API Reference
~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   toplevel
   types
   eclass
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
