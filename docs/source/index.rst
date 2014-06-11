Blaze
=====

Blaze provides a uniform and familiar interface to diverse computation and data
storage.  Blaze extends the usability of NumPy and Pandas to diverse
computational systems, enabling easy access to out-of-core, streaming, and
distributed data.

Blaze allows analysts and scientists to productively write robust and efficient
code, without getting bogged down in the details of how to distribute
computation, or convert data between databases, formats, proprietary data
warehouses, and other silos.


Core
----

The core of Blaze consists of

*   A type system to express data types and layouts
*   A familiar user interface and symbolic expression system
*   A set of interfaces to common data formats
*   A set of interfaces to powerful computational engines

Blaze depends on and exposes the hard work of countless projects.


Under Development
-----------------

Please be aware that Blaze is under development. The documentation and
implementation this project have gone through many changes, and there are many
places where the implementation has not been properly brought into sync with
this documentation.

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
