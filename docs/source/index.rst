
.. image:: https://raw.github.com/ContinuumIO/blaze/master/docs/source/svg/blaze_med.png
   :align: center

**Blaze** translates a subset of modified NumPy and Pandas-like syntax to
databases and other computing systems.  Blaze allows Python users a familiar
interface to query data living in other data storage systems.

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

Presentations
-------------

* `See previous presentations about Blaze`_


Index
~~~~~

User facing

.. toctree::
   :maxdepth: 1

   overview
   install
   quickstart
   queries
   rosetta
   uri
   csv
   sql
   ooc
   server
   datashape
   api

Internal

.. toctree::
   :maxdepth: 1

   expressions
   backends
   interactivity
   dev_workflow
   expr-compute-dev
   computation

Older Versions
~~~~~~~~~~~~~~

Older versions of these documents can be found here_.

.. _here: ../

.. _`See previous presentations about Blaze`: http://blaze.pydata.org/presentations
