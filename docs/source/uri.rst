===========
URI strings
===========

Blaze uses strings to specify data resources.  This is purely for ease of use.

Example
-------

Interact with a set of CSV files or a SQL database

.. code-block:: python

   >>> from blaze import *
   >>> from blaze.utils import example
   >>> t = data(example('accounts_*.csv'))
   >>> t.peek()
      id      name  amount
   0   1     Alice     100
   1   2       Bob     200
   2   3   Charlie     300
   3   4       Dan     400
   4   5     Edith     500

   >>> t = data('sqlite:///%s::iris' % example('iris.db'))
   >>> t.peek()
       sepal_length  sepal_width  petal_length  petal_width      species
   0            5.1          3.5           1.4          0.2  Iris-setosa
   1            4.9          3.0           1.4          0.2  Iris-setosa
   2            4.7          3.2           1.3          0.2  Iris-setosa
   3            4.6          3.1           1.5          0.2  Iris-setosa
   4            5.0          3.6           1.4          0.2  Iris-setosa
   5            5.4          3.9           1.7          0.4  Iris-setosa
   6            4.6          3.4           1.4          0.3  Iris-setosa
   7            5.0          3.4           1.5          0.2  Iris-setosa
   8            4.4          2.9           1.4          0.2  Iris-setosa
   9            4.9          3.1           1.5          0.1  Iris-setosa
   ...


Migrate CSV files into a SQL database

.. code-block:: python

   >>> from odo import odo
   >>> odo(example('iris.csv'), 'sqlite:///myfile.db::iris') # doctest: +SKIP
   Table('iris', MetaData(bind=Engine(sqlite:///myfile.db)), ...)

What sorts of URIs does Blaze support?
--------------------------------------

* Paths to files on disk, including the following extensions
    * ``.csv``
    * ``.json``
    * ``.csv.gz/json.gz``
    * ``.hdf5`` (uses ``h5py``)
    * ``.hdf5::/datapath``
    * ``hdfstore://filename.hdf5`` (uses special ``pandas.HDFStore`` format)
    * ``.bcolz``
    * ``.xls(x)``
* SQLAlchemy strings like the following
    * ``sqlite:////absolute/path/to/myfile.db::tablename``
    * ``sqlite:////absolute/path/to/myfile.db``  (specify a particular table)
    * ``postgresql://username:password@hostname:port``
    * ``impala://hostname`` (uses ``impyla``)
    * *anything supported by SQLAlchemy*
* MongoDB Connection strings of the following form
    * ``mongodb://username:password@hostname:port/database_name::collection_name``
* Blaze server strings of the following form
    * ``blaze://hostname:port``  (port defaults to 6363)

In all cases when a location or table name is required in addition to the traditional URI (e.g. a data path within an HDF5 file or a Table/Collection name within a database) then that information follows on the end of the URI after a separator of two colons ``::``.

How it works
------------
Blaze depends on the `Odo <https://github.com/blaze/odo>`_ library to handle URIs.
URIs are managed through the ``resource`` function which is dispatched based on regular expressions.  For example a simple resource function to handle ``.json`` files might look like the following (although Blaze's actual solution is a bit more comprehensive):

.. code-block:: python

   from blaze import resource
   import json

   @resource.register('.+\.json')
   def resource_json(uri):
       with open(uri):
           data = json.load(uri)
       return data


Can I extend this to my own types?
----------------------------------

Absolutely.  Import and extend ``resource`` as shown in the "How it works" section.  The rest of Blaze will pick up your change automatically.
