
.. image:: https://raw.github.com/ContinuumIO/blaze/master/docs/source/svg/blaze_med.png
   :align: center

**Blaze** translates a subset of modified NumPy and Pandas-like syntax to
databases and other computing systems.  Blaze allows Python users a familiar
interface to query data living in other data storage systems.


Ecosystem
---------

Several projects have come out of Blaze development other than the Blaze
project itself.

* Blaze: Translates NumPy/Pandas-like syntax to databases.  High level user
  interaction and analysis.
* Into_: Migrates data between formats.
* Dask.array_: Multi-core / on-disk NumPy arrays

The rest of this documentation is just about the Blaze project itself.  See the
pages linked to above for ``into`` or ``dask.array``.


Blaze
-----

Blaze is a high-level user interface for databases and array computing systems.
It consists of the following components:

*   A symbolic expression system to describe and reason about analytic queries
*   A set of interpreters from that query system to various databases /
   computational engines

This architecture allows a single Blaze code to run against several
computational backends.  Blaze interacts rapidly with the user and only
communicates with the database when necessary.  Blaze is also able to analyze
and optimize queries to improve the interactive experience.


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
.. _Into: http://into.readthedocs.org/
.. _Dask.array: http://dask.readthedocs.org/
