======
Server
======

Blaze provides uniform access to a variety of common data formats.  Blaze
Server builds off of this uniform interface to host data remotely through a
JSON web API.

Setting up a Blaze Server
=========================

To demonstrate the use of the Blaze server we serve the iris csv file.

.. code-block:: python

   >>> # Server code, run this once.  Leave running.

   >>> from blaze import *
   >>> csv = CSV('blaze/examples/data/iris.csv')
   >>> Data(csv)
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


Then we host this publicly on port ``6363``


.. code-block:: python

   from blaze.server import Server
   server = Server(csv)
   server.run(host='0.0.0.0', port=6363)

A Server is the following

1.  A dataset that blaze understands or dictionary of such datasets
2.  A Flask_ app.

With this code our machine is now hosting our CSV file through a
web-application on port 6363.  We can now access our CSV file, through Blaze,
as a service from a variety of applications.

Interacting with the Web Server from the Client
===============================================

Computation is now available on this server at
``hostname:6363/compute.json``.  To communicate the computation to be done
we pass Blaze expressions in JSON format through the request.  See the examples
below.

Using ``curl``
--------------

We can use standard command line tools to interact with this web service::

   $ curl \
       -H "Content-Type: application/json" \
       -d '{"expr": {"op": "Field", "args": [":leaf", "species"]}}' \
       localhost:6363/compute.json

   {
     "data": [
         "Iris-setosa",
         "Iris-setosa",
         ...
         ],
     "datashape": "var * {species: string}",
   }

   $ curl \
       -H "Content-Type: application/json" \
       -d  '{"expr": {"op": "sum", \
                      "args": [{"op": "Field", \
                                "args": [":leaf", "petal_Length"]}]}}' \
       localhost:6363/compute.json

   {
     "data": 563.8000000000004,
     "datashape": "{petal_length_sum: float64}",
   }

These queries deconstruct the Blaze expression as nested JSON.  The ``":leaf"``
string is a special case pointing to the base data.  Constructing these queries
can be difficult to do by hand, fortunately Blaze can help you to build them.


Using the Python Requests Library
---------------------------------

First we repeat the same experiment as before, this time using the Python
``requests`` library instead of the command line tool ``curl``.

.. code-block:: python

   # Client code, run this in a separate process from the Server

   import json
   import requests

   query = {'expr': {'op': 'sum',
                     'args': [{'op': 'Field',
                               'args': [':leaf', 'petal_length']}]}}

   r = requests.get('http://localhost:6363/compute.json',
                   data=json.dumps(query),
                   headers={'Content-Type': 'application/json'})

   json.loads(r.content)

  {u'data': 563.8000000000004,
   u'datashape': u'{petal_length_sum: float64}'}

Now we use Blaze to generate the query programmatically

.. code-block:: python

   >>> from blaze import *

   >>> # Build a Symbol like our served iris data
   >>> dshape= "var * {sepal_length: float64, sepal_width: float64, petal_length: float64, petal_width: float64, species: string}"  # matching schema to csv file
   >>> t = symbol('t', dshape)
   >>> expr = t.petal_length.sum()

   >>> from blaze.server import to_tree

   >>> d = to_tree(expr, names={t: ':leaf'})

   >>> query = {'expr': d}
   >>> query  # doctest: +SKIP
   {'expr': {'args': [{'args': [':leaf', 'petal_length'],
                         'op': 'Field'},
                      [0],
                      False],
               'op': 'sum'}}

Alternatively we build a query to grab a single column

.. code-block:: python

   >>> to_tree(t.species, names={t: ':leaf'})  # doctest: +SKIP
   {'args': [':leaf', 'species'], 'op': 'Field'}


Fully Interactive Python-to-Python Remote work
----------------------------------------------

Alternatively we can use this API to have one Blaze process control another.
Given our iris web server we can use Blaze on the client to drive the server to
do work for us

.. code-block:: python

   # Client code, run this in a separate process from the Server

   >>> from blaze import Data, by
   >>> t = Data('blaze://localhost:6363')  # doctest: +SKIP

   >>> t  # doctest: +SKIP
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

   >>> by(t.species, min=t.petal_length.min(),
   ...               max=t.petal_length.max())  # doctest: +SKIP
              species  max  min
   0   Iris-virginica  6.9  4.5
   1      Iris-setosa  1.9  1.0
   2  Iris-versicolor  5.1  3.0

We interact on the client machine through the data object but computations on
this object cause communications through the web API, resulting in seemlessly
interactive remote computation.


Advanced Use
------------

Blaze servers may host any data that Blaze understands from a single integer

.. code-block:: python

   >>> server = Server(1)

To a dictionary of several heterogeneous datasets

.. code-block:: python

   >>> server = Server({'my-dataframe': df,
   ...                  'iris': resource('iris.csv'),
   ...                  'baseball': resource('sqlite:///baseball-statistics.db')})  # doctest: +SKIP

A variety of hosting options are available through the Flask_ project

::

   >>> help(server.app.run)  # doctest: +SKIP
   Help on method run in module flask.app:

   run(self, host=None, port=None, debug=None, **options) method of  flask.app.Flask instance
   Runs the application on a local development server.  If the
   :attr:`debug` flag is set the server will automatically reload
   for code changes and show a debugger in case an exception happened.

   ...


Caching
-------

Caching results on frequently run queries may significantly improve user
experience in some cases.  One may wrap a Blaze server in a traditional
web-based caching system like memcached or use a data centric solution.

The Blaze ``CachedDataset`` might be appropriate in some situations.  A cached
dataset holds a normal dataset and a ``dict`` like object.

.. code-block:: python

   >>> dset = {'my-dataframe': df,
   ...         'iris': resource('iris.csv'),
   ...         'baseball': resource('sqlite:///baseball-statistics.db')} # doctest: +SKIP

   >>> from blaze.cached import CachedDataset  # doctest: +SKIP
   >>> cached = CachedDataset(dset, cache=dict())  # doctest: +SKIP

Queries and results executed against a cached dataset are stored in the cache
(here a normal Python ``dict``) for fast future access.

If accumulated results are likely to fill up memory then other, on-disk
dict-like structures can be used like Shove_ or Chest_.

.. code-block:: python

   >>> from chest import Chest  # doctest: +SKIP
   >>> cached = CachedDataset(dset, cache=Chest())  # doctest: +SKIP

These cached objects can be used anywhere normal objects can be used in Blaze,
including an interactive (and now performance cached) ``Data`` object

.. code-block:: python

   >>> d = Data(cached)  # doctest: +SKIP

or a Blaze server

.. code-block:: python

   >>> server = Server(cached)  # doctest: +SKIP


Flask Blueprint
---------------

If you would like to use the blaze server endpoints from within another flask
application, you can register the blaze api blueprint with your app. For
example:

.. code-block:: python

   >>> from blaze.server import api  # doctest: +SKIP
   >>> app.register_blueprint(api)  # doctest: +SKIP
   >>> with app.app_context():  # doctest: +SKIP
   >>>     flask.g.data = dset  # doctest: +SKIP
   >>>     app.run()  # doctest: +SKIP


When using the blaze flask blueprint, be sure to register your dataset in the
flask application context ``flask.g`` so that it is available to the API
endpoints.


Conclusion
==========

Because this process builds off Blaze expressions it works equally well for data
stored in any format on which Blaze is trained, including in-memory DataFrames,
SQL/Mongo databases, or even Spark clusters.


.. _Flask : http://flask.pocoo.org/docs/0.10/quickstart/#a-minimal-application
.. _Shove : https://pypi.python.org/pypi/shove/0.5.6
.. _Chest : https://github.com/mrocklin/chest
