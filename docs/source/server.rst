======
Server
======

Blaze provides uniform access to a variety of common data formats.  Blaze
Server builds off of this uniform interface to host data remotely through a
simple REST-ful API.

Setting up a Blaze Server
=========================

To demonstrate the use of the Blaze server we serve the iris csv file.

.. code-block:: python

   >>> # Server code, run this once.  Leave running.

   >>> from blaze import *
   >>> csv = CSV('examples/data/iris.csv')
   >>> csv.schema
   dshape("{ sepal_length : ?float64, sepal_width : ?float64, petal_length :
   ?float64, petal_width : ?float64, species : string }")

   >>> Table(csv)
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


Then we host this under the name ``'iris'`` and serve publicly on port
``5000``


.. code-block:: python

   >>> from blaze.server import Server
   >>> server = Server({'iris': csv})
   >>> server.app.run(host='0.0.0.0', port=5000)

A Server is the following

1.  A mapping of names to datasets
2.  A `Flask <http://flask.pocoo.org/>`_ app.

With this code our machine is now hosting our CSV file through a
web-application on port 5000.  We can now access our CSV file, through Blaze,
as a service from a variety of applications.

Interacting with the Web Server from the Client
===============================================

Computation is now available on this server at
``hostname:5000/compute/iris.json``.  To communicate the computation to be done
we pass Blaze expressions in JSON format through the request.  See the examples
below.

Using ``curl``
--------------

We can use standard command line tools to interact with this web service::

   $ curl \
       -H "Content-Type: application/json" \
       -d '{"expr": {"op": "Column", "args": ["iris", "species"]}}' \
       localhost:5000/compute/iris.json

   {
     "data": [
         "setosa",
         "setosa",
         ...
         ],
     "datashape": "var * { species : string }",
     "name": "iris"
   }

   $ curl \
       -H "Content-Type: application/json" \
       -d  '{"expr": {"op": "sum", \
                      "args": [{"op": "Column", \
                                 "args": ["iris", "petal_Length"]}]}}' \
       localhost:5000/compute/iris.json

   {
     "data": 563.8000000000004,
     "datashape": "{ petal_length_sum : ?float64 }",
     "name": "iris"
   }


Constructing these queries can be difficult to do by hand, fortunately Blaze
can help you to build them.


Using the Python Requests Library
---------------------------------

First we repeat the same experiment as before, this time using the Python
``requests`` library instead of the command line tool ``curl``.

.. code-block:: python

   >>> # Client code, run this in a separate process from the Server

   >>> import json
   >>> import requests

   >>> query = {'expr': {'op': 'sum',
   ...                   'args': [{'op': 'Column',
   ...                             'args': ['iris', 'petal_length']}]}}

   >>> r = requests.get('http://localhost:5000/compute/iris.json',
   ...                 data=json.dumps(query),
   ...                 headers={'Content-Type': 'application/json'})

   >>> json.loads(r.content)
   {u'data': 563.8000000000004,
    u'datashape': u'{ petal_length_sum : ?float64 }',
    u'name': u'iris'}

Now we use Blaze to generate the query programmatically

.. code-block:: python

   >>> from blaze import *

   >>> schema = "{ sepal_length : ?float64, sepal_width : ?float64, petal_length : ?float64, petal_width : ?float64, species : string }"  # matching schema to csv file

   >>> t = TableSymbol('t', schema)
   >>> expr = t.petal_length.sum()

   >>> from blaze.server import to_tree

   >>> d = to_tree(expr, names={t: 'iris'})
   >>> d
   {'op': 'sum', 'args': [{'args': ['iris', 'petal_length'], 'op': 'Column'}]}

   >>> query = {'expr': d}

Alternatively we build a query to grab a single column

.. code-block:: python

   >>> to_tree(t.species, names={t: 'iris'})
   {'op': 'Column', 'args': ['iris', 'species']}

Fully Interactive Python-to-Python Remote work
----------------------------------------------

Alternatively we can use this API to have one Blaze process control another.
Given our iris web server we can use Blaze on the client to drive the server to
do work for us

.. code-block:: python

   >>> # Client code, run this in a separate process from the Server

   >>> from blaze import *
   >>> from blaze.server import ExprClient
   >>> ec = ExprClient('http://localhost:5000', 'iris')

   >>> t = Table(ec)
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

   >>> by(t, t.species, min=t.petal_length.min(), max=t.petal_length.max())
              species  max  min
   0   Iris-virginica  6.9  4.5
   1      Iris-setosa  1.9  1.0
   2  Iris-versicolor  5.1  3.0

We interact on the client machine through the ``ExprClient`` data object but
computations on this object cause communications through the web API, resulting
in seemlessly interactive remote computation.

Conclusion
==========

Because this process builds off Blaze expressions it works
equally well for data stored in any format on which Blaze is trained, including in-memory DataFrames, SQL/Mongo databases, or even Spark clusters.
