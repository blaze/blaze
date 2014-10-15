===========
URI strings
===========

Blaze uses strings to specify data resources.  This is purely for ease of use.

Example
-------

Interact with a set of CSV files or a SQL database

.. code-block:: python

   >>> from blaze import *
   >>> t = Data('blaze/examples/data/account_*.csv')
   >>> t


   >>> t = Data('sqlite:///blaze/examples/data/iris.db::iris')
   >>> t

Migrate CSV files into a SQL database

.. code-block:: python

   >>> into('sqlite:///myfile.db::iris', 'blaze.examples/data/iris.csv') # doctest: +SKIP
   <>

What sorts of URI's does Blaze support?
---------------------------------------

* Paths to files on disk, including the following extensions
  * ``.csv``
  * ``.json``
  * ``.csv.gz/json.gz``
  * ``.hdf5::/datapath`` (uses ``h5py``)
  * ``.h5::/datapath`` (uses ``PyTables``)
  * ``.bcolz``
  * ``.xls(x)``
* SQLAlchemy strings like the following
  * ``sqlite:////absolute/path/to/myfile.db::tablename``
  * ``postgresql://username:password@hostname:port::tablename``
  * ``impala://hostname::tablename`` (uses ``impyla``)
  * *anything supported by SQLAlchemy*
* MongoDB Connection strings of the following form
  * ``mongodb://username:password@hostname:port/database_name::collection_name``
* Blaze server strings of the following form
  * ``blaze://hostname:port::dataset_name``  (port defaults to 6363)

In all cases when a location or table name is required in addition to the traditional URI (e.g. a data path within an HDF5 file or a Table/Collection name within a database) then that information follows on the end of the URI after a separator of two colons ``::``.

How it works
------------

URIs are managed through the ``resource`` function which is dispatched based on regular expressions.  For example a simple resource function to handle ``.json`` files might look like the following (although Blaze's actual solution is a bit more comprehensive):

.. code-block:: python

   from blaze import resource
   import json

   @resource.register('.+\.json')
   def resource_json(uri):
       with open(uri):
           data = json.load(uri)
       return data


Where does this work in Blaze?
------------------------------

URIs are supported through the resource function internally.  Other user-facing functions use resource if they are given a string.  So far this includes the following

*  ``Data`` as shown at the top of this page
*  ``into`` as shown at the top of this page


Can I extend this to my own types?
----------------------------------

Absolutely.  Import and extend ``resource`` as shown in the "How it works" section.  The rest of Blaze will pick up your change automatically.
