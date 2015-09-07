
.. image:: svg/blaze_med.png
   :align: center

**Blaze** translates a subset of modified NumPy and Pandas-like syntax to
databases and other computing systems.  Blaze allows Python users a familiar
interface to query data living in other data storage systems.   Blaze is sponsored primarily by 
`Continuum Analytics <http://www.continuum.io>`_, and a
`DARPA XDATA <http://www.darpa.mil/program/XDATA>`_ grant.


Ecosystem
---------

Several projects have come out of Blaze development other than the Blaze
project itself.

* Blaze: Translates NumPy/Pandas-like syntax to data computing systems (e.g. database, in-memory, distributed-computing) including data gate-way server that can sit on computing machines near the data --- moving the expression to the data. 

  Blaze presents a pleasant and familiar interface to us regardless of
  what computational solution or database we use (e.g. spark, impala, SQL databases, 
  No-SQL data-stores, raw-files).  It mediates our interaction with files, data structures, 
  and databases, optimizing and translating our query as appropriate to provide a smooth and interactive
  session.   It allows the data scientists and analyst to write their queries in a unified way that does 
  not have to change because the data is stored in another format or a different data-store. 
  It also provides a server-component that allows URIs to be used to easily serve views on data and refer to Data 
  remotely in local scripts, queries, and programs.

* Odo_: Migrates data between formats.

  Odo moves data between formats (CSV, JSON, databases) and locations
  (local, remote, HDFS) efficiently and robustly with a dead-simple interface
  by leveraging a sophisticated and extensible network of conversions.

* Dask.array_: Multi-core / on-disk NumPy arrays
* Dask.dataframe_ : Multi-core / on-disk Pandas data-frames

  Dask.arrays provide blocked algorithms on top of NumPy to handle
  larger-than-memory arrays and to leverage multiple cores.  They are a
  drop-in replacement for a commonly used subset of NumPy algorithms.
  
  Dask.dataframes provide blocked algorithms on top of Pandas to handle 
  larger-than-memory data-frames and to leverage multiple cores.   They 
  are a drop-in replacement for a subset of Pandas use-cases.
  
  Dask also has a general "Bag" type and a way to build "task graphs"
  using simple decorators as well as nascent distributed schedulers in 
  addition to the multi-core and multi-threaded schedulers.

* DyND_: In-memory dynamic arrays

  DyND is a dynamic ND-array library like NumPy.  It supports variable length
  strings, ragged arrays, and GPUs.  It is a standalone C++ codebase with
  Python bindings.  Generally it is more extensible than NumPy but also less
  mature.

These projects are mutually independent.  The rest of this documentation is
just about the Blaze project itself.  See the pages linked to above for ``odo``
or ``dask.array``.    Other projects that have spun out of or are linked to 
Blaze efforts include:
    * bcolz_ : http://bcolz.blosc.org/ -- A columnar data container that can be compressed.
    * castra_ : https://github.com/Blosc/castra -- partitioned storage system based on blosc compression.
    


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
   split-apply-combine
   rosetta-pandas
   rosetta-sql
   uri
   csv
   sql
   ooc
   server
   datashape
   what-blaze-isnt
   api
   releases
   people
   legal

Internal

.. toctree::
   :maxdepth: 1

   expr-design
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

.. _`See previous presentations about Blaze`: _static/presentations/index.html
.. _`See previous blog posts about Blaze`: http://continuum.io/blog/tags/blaze
.. _Odo: http://odo.pydata.org/
.. _Dask.array: http://dask.pydata.org/
.. _Dask.dataframe: http://dask.pydata.org
.. _DyND: https://github.com/libdynd/libdynd
