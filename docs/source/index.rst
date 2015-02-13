
.. image:: https://raw.github.com/ContinuumIO/blaze/master/docs/source/svg/blaze_med.png
   :align: center

**Blaze** translates a subset of modified NumPy and Pandas-like syntax to
databases and other computing systems.  Blaze allows Python users a familiar
interface to query data living in other data storage systems.


Ecosystem
---------

Several projects have come out of Blaze development other than the Blaze
project itself.

* Blaze: Translates NumPy/Pandas-like syntax to systems like databases.

  Blaze presents a pleasant and familiar interface to us regardless of
  what computational solution or database we use.  It mediates our
  interaction with files, data structures, and databases, optimizing and
  translating our query as appropriate to provide a smooth and interactive
  session.

* Into_: Migrates data between formats.

  Into moves data between formats (CSV, JSON, databases) and locations
  (local, remote, HDFS) efficiently and robustly with a dead-simple interface
  by leveraging a sophisticated and extensible network of conversions.

* Dask.array_: Multi-core / on-disk NumPy arrays

  Dask.arrays provide blocked algorithms on top of NumPy to handle
  larger-than-memory arrays and to leverage multiple cores.  They are a
  drop-in replacement for a commonly used subset of NumPy algorithms.


The rest of this documentation is just about the Blaze project itself.  See the
pages linked to above for ``into`` or ``dask.array``.


Blaze
-----

Blaze is a high-level user interface for databases and array computing systems.
It consists of the following components:

*   A symbolic expression system to describe and reason about analytic queries
*   A set of interpreters from that query system to various databases / computational engines

This architecture allows a single Blaze code to run against several
computational backends.  Blaze interacts rapidly with the user and only
communicates with the database when necessary.  Blaze is also able to analyze
and optimize queries to improve the interactive experience.


Presentations
-------------

* `See previous presentations about Blaze`_
* `See previous blog posts about Blaze`_


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
.. _`See previous blog posts about Blaze`: http://continuum.io/blog/tags/blaze
.. _Into: http://into.readthedocs.org/
.. _Dask.array: http://dask.readthedocs.org/
