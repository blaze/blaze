
.. image:: https://raw.github.com/ContinuumIO/blaze/master/docs/source/svg/blaze_med.png
   :align: center

**Blaze** provides a familiar interface to various computational resources.

Blaze extends the usability of NumPy and Pandas to distributed and
out-of-core computing.  Blaze provides an interface similar to that of the
NumPy ND-Array or Pandas DataFrame but maps these familiar interfaces onto a
variety of other computational engines like Postgres or Spark.


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
   :maxdepth: 1

   install
   overview
   quickstart
   data
   expressions
   backends
   datashape
   server
   api
   dev_workflow
   expr-compute-dev

Older Versions
~~~~~~~~~~~~~~

Older versions of these documents can be found here_.

.. _here: ../
