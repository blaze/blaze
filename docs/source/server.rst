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


Then we host this under the name ``'iris'`` and serve publicly on port
``6363``


.. code-block:: python

   from blaze.server import Server
   server = Server({'iris': csv})
   server.run(host='0.0.0.0', port=6363)

A Server is the following

1.  A mapping of names to datasets
2.  A `Flask <http://flask.pocoo.org/docs/0.10/quickstart/#a-minimal-application>`_ app.

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
       -d '{"expr": {"op": "Field", "args": ["iris", "species"]}}' \
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
                                "args": ["iris", "petal_Length"]}]}}' \
       localhost:6363/compute.json

   {
     "data": 563.8000000000004,
     "datashape": "{petal_length_sum: float64}",
   }


Constructing these queries can be difficult to do by hand, fortunately Blaze
can help you to build them.


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
                               'args': ['iris', 'petal_length']}]}}

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
   >>> t = Symbol('t', dshape)
   >>> expr = t.petal_length.sum()

   >>> from blaze.server import to_tree

   >>> d = to_tree(expr, names={t: 'iris'})

   >>> query = {'expr': d}

Alternatively we build a query to grab a single column

.. code-block:: python

   >>> b = to_tree(t.species, names={t: 'iris'})

Fully Interactive Python-to-Python Remote work
----------------------------------------------

Alternatively we can use this API to have one Blaze process control another.
Given our iris web server we can use Blaze on the client to drive the server to
do work for us

.. code-block:: python

   # Client code, run this in a separate process from the Server

   from blaze import *
   c = Client('http://localhost:6363', 'iris')

   t = Data(c)
   t

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

   by(t.species, min=t.petal_length.min(),
                 max=t.petal_length.max())
              species  max  min
   0   Iris-virginica  6.9  4.5
   1      Iris-setosa  1.9  1.0
   2  Iris-versicolor  5.1  3.0

We interact on the client machine through the ``Client`` data object but
computations on this object cause communications through the web API, resulting
in seemlessly interactive remote computation.

Conclusion
==========

Because this process builds off Blaze expressions it works
equally well for data stored in any format on which Blaze is trained, including in-memory DataFrames, SQL/Mongo databases, or even Spark clusters.
