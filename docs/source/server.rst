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
   >>> from blaze.utils import example
   >>> csv = CSV(example('iris.csv'))
   >>> data(csv).peek()
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


Then we host this publicly on port 6363


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

Serving Data from the Command Line
==================================

Blaze ships with a command line tool called ``blaze-server`` to serve up data
specified in a YAML file.

.. note::

   To use the YAML specification feature of Blaze server please install
   the :mod:`pyyaml` library. This can be done easily with ``conda``:

   .. code-block:: sh

      conda install pyyaml

YAML Specification
------------------

The structure of the specification file is as follows:

  .. code-block:: yaml

     name1:
       source: path or uri
       dshape: optional datashape
     name2:
       source: path or uri
       dshape: optional datashape
     ...
     nameN:
       source: path or uri
       dshape: optional datashape

.. note::

  When ``source`` is a directory, Blaze will recurse into the directory tree
  and call ``odo.resource`` on the leaves of the tree.

Here's an example specification file:

  .. code-block:: yaml

     iriscsv:
       source: ../examples/data/iris.csv
     irisdb:
       source: sqlite:///../examples/data/iris.db
     accounts:
       source: ../examples/data/accounts.json.gz
       dshape: "var * {name: string, amount: float64}"


The previous YAML specification will serve the following dictionary:

  .. code-block:: python

     >>> from odo import resource
     >>> resources = {
     ...  'iriscsv': resource('../examples/data/iris.csv'),
     ...  'irisdb': resource('sqlite:///../examples/data/iris.db'),
     ...  'accounts': resource('../examples/data/accounts.json.gz',
     ...                       dshape="var * {name: string, amount: float64}")
     ... }


The only required key for each named data source is the ``source`` key, which
is passed to ``odo.resource``. You can optionally specify a ``dshape``
parameter, which is passed into ``odo.resource`` along with the ``source`` key.

Advanced YAML usage
-------------------

If ``odo.resource`` requires extra keyword arguments for a particular resource
type and they are provided in the YAML file, these will be forwarded on to the
``resource`` call.

If there is an ``imports`` entry for a resource whose value is a list of module
or package names, Blaze server will ``import`` each of these modules or
packages before calling ``resource``.

For example:

  .. code-block:: yaml

     name1:
         source: path or uri
         dshape: optional datashape
         kwarg1: extra kwarg
         kwarg2: etc.
     name2:
         source: path or uri
         imports: ['mod1', 'pkg2']

For this YAML file, Blaze server will pass on ``kwarg1=...`` and ``kwarg2=...``
to the ``resource()`` call for ``name1`` in addition to the ``dshape=...``
keyword argument.

Also, before calling ``resource`` on the ``source`` of ``name2``, Blaze server
will first execute an ``import mod1`` and ``import pkg2`` statement.

Command Line Interface
----------------------

  1. UNIX

    .. code-block:: shell

       # YAML file specifying resources to load and optionally their datashape
       $ cat example.yaml
       iriscsv:
         source: ../examples/data/iris.csv
       irisdb:
         source: sqlite:///../examples/data/iris.db
       accounts:
         source: ../examples/data/accounts.json.gz
         dshape: "var * {name: string, amount: float64}"

       # serve data specified in a YAML file and follow symbolic links
       $ blaze-server example.yaml --follow-links

       # You can also construct a YAML file from a heredoc to pipe to blaze-server
       $ cat <<EOF
       datadir:
         source: /path/to/data/directory
       EOF | blaze-server

  2. Windows

    .. code-block:: powershell

       # If you're on Windows you can do this with powershell
       PS C:\> @'
       datadir:
         source: C:\path\to\data\directory
       '@ | blaze-server


Interacting with the Web Server from the Client
===============================================

Computation is now available on this server at
``localhost:6363/compute.json``. To communicate the computation to be done
we pass Blaze expressions in JSON format through the request.  See the examples
below.

Fully Interactive Python-to-Python Remote work
----------------------------------------------

The highest level of abstraction and the level that most will probably want to
work at is interactively sending computations to a Blaze server process from a
client.

We can use Blaze server to have one Blaze process control another.  Given our
iris web server we can use Blaze on the client to drive the server to do work
for us

.. code-block:: python

   # Client code, run this in a separate process from the Server

   >>> from blaze import data, by
   >>> t = data('blaze://localhost:6363')  # doctest: +SKIP

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

The blaze server and client can be configured to support various serialization
formats. These formats are exposed in the :mod:`blaze.server` module. The server
and client must both be told to use the same serialization format.
For example:

.. code-block:: python

    # Server setup.
    >>> from blaze import Server
    >>> from blaze.server import msgpack_format, json_format
    >>> Server(my_data, formats=(msgpack_format, json_format).run()  # doctest: +SKIP

    # Client code, run this in a separate process from the Server
    >>> from blaze import Client, data
    >>> from blaze.server import msgpack_format, json_format
    >>> msgpack_client = data(Client('localhost', msgpack_format))  # doctest: +SKIP
    >>> json_client = data(Client('localhost', json_format))  # doctest +SKIP

In this example, ``msgpack_client`` will make requests to the
``/compute.msgpack`` endpoint and will send and receive data using the msgpack
protocol; however, the ``json_client`` will make requests to the
``/compute.json`` endpoint and will send and receive data using the json
protocol.

Using the Python Requests Library
---------------------------------

Moving down the stack, we can interact at the HTTP request level with Blaze
serer using the ``requests`` library.

.. code-block:: python

   # Client code, run this in a separate process from the Server

   >>> import json
   >>> import requests
   >>> query = {'expr': {'op': 'sum',
   ...                   'args': [{'op': 'Field',
   ...                             'args': [':leaf', 'petal_length']}]}}
   >>> r = requests.get('http://localhost:6363/compute.json',
   ...                  data=json.dumps(query),
   ...                  headers={'Content-Type': 'application/vnd.blaze+json'})  # doctest: +SKIP
   >>> json.loads(r.content)  # doctest: +SKIP
   {u'data': 563.8000000000004,
    u'names': ['petal_length_sum'],
    u'datashape': u'{petal_length_sum: float64}'}

Now we use Blaze to generate the query programmatically

.. code-block:: python

   >>> from blaze import symbol
   >>> from blaze.server import to_tree
   >>> from pprint import pprint

   >>> # Build a Symbol like our served iris data
   >>> dshape = """var * {
   ...     sepal_length: float64,
   ...     sepal_width: float64,
   ...     petal_length: float64,
   ...     petal_width: float64,
   ...     species: string
   ... }"""  # matching schema to csv file
   >>> t = symbol('t', dshape)
   >>> expr = t.petal_length.sum()
   >>> d = to_tree(expr, names={t: ':leaf'})
   >>> query = {'expr': d}
   >>> pprint(query)
   {'expr': {'args': [{'args': [':leaf', 'petal_length'], 'op': 'Field'},
                      [0],
                      False],
             'op': 'sum'}}

Alternatively we build a query to grab a single column

.. code-block:: python

   >>> pprint(to_tree(t.species, names={t: ':leaf'}))
   {'args': [':leaf', 'species'], 'op': 'Field'}


Using ``curl``
--------------

In fact, any tool that is capable of sending requests to a server is able to
send computations to a Blaze server.

We can use standard command line tools such as ``curl`` to interact with the
server::

   $ curl \
       -H "Content-Type: application/vnd.blaze+json" \
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
       -H "Content-Type: application/vnd.blaze+json" \
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

Adding Data to the Server
-------------------------

Data resources can be added to the server from the client by sending a resource
URI to the server. The data initially on the server must have a dictionary-like
interface to be updated.

.. code-block:: python

   >>> from blaze.utils import example
   >>> query = {'accounts': example('accounts.csv')}
   >>> r = requests.get('http://localhost:6363/add',
   ...                  data=json.dumps(query),
   ...                  headers={'Content-Type': 'application/vnd.blaze+json'})  # doctest: +SKIP


Advanced Use
------------

Blaze servers may host any data that Blaze understands from a single integer

.. code-block:: python

   >>> server = Server(1)

To a dictionary of several heterogeneous datasets

.. code-block:: python

   >>> server = Server({
   ...     'my-dataframe': df,
   ...     'iris': resource('iris.csv'),
   ...     'baseball': resource('sqlite:///baseball-statistics.db')
   ... })  # doctest: +SKIP

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
(here a normal Python :class:`dict`) for fast future access.

If accumulated results are likely to fill up memory then other, on-disk
``dict``-like structures can be used like Shove_ or Chest_.

.. code-block:: python

   >>> from chest import Chest  # doctest: +SKIP
   >>> cached = CachedDataset(dset, cache=Chest())  # doctest: +SKIP

These cached objects can be used anywhere normal objects can be used in Blaze,
including an interactive (and now performance cached) ``data`` object

.. code-block:: python

   >>> d = data(cached)  # doctest: +SKIP

or a Blaze server

.. code-block:: python

   >>> server = Server(cached)  # doctest: +SKIP


Flask Blueprint
---------------

If you would like to use the blaze server endpoints from within another flask
application, you can register the blaze API blueprint with your application.
For example:

.. code-block:: python

   >>> from blaze.server import api, json_format
   >>> my_app.register_blueprint(api, data=my_data, formats=(json_format,))  # doctest: +SKIP


When registering the API, you must pass the data that the API endpoints will
serve.
You must also pass an iterable of serialization format objects that the server
will respond to.


Profiling
---------

The blaze server allows users and server administrators to profile computations
run on the server. This allows developers to better understand the performance
profile of their computations to better tune their queries or the backend code
that is executing the query. This profiling will also track the time spent in
serializing the data.

By default, blaze servers will not allow profiling. To enable profiling on the
blaze server, pass ``allow_profiler=True`` to the
:class:`~blaze.server.server.Server` object. Now when we try to compute against
this server, we may pass ``profile=True`` to ``compute``. For example:


.. code-block:: python

   >>> client = Client(...)  # doctest: +SKIP
   >>> compute(expr, client, profile=True)  # doctest: +SKIP


After running the above code, the server will have written a new pstats file
containing the results of the run. This fill will be found at:
``profiler_output/<md5>/<timestamp>``. We use the md5 hash of the str of the
expression so that users can more easily track down their stats
information. Users can find the hash of their expression with
:func:`~blaze.server.server.expr_md5`.

The profiler output directory may be configured with the ``profiler_output``
argument to the :class:`~blaze.server.server.Server`.

Clients may also request that the profiling data be sent back in the response so
that analysis may happen on the client. To do this, we change our call to
compute to look like:

.. code-block:: python

   >>> from io import BytesIO  # doctest: +SKIP
   >>> buf = BytesIO()  # doctest: +SKIP
   >>> compute(expr, client, profile=True, profiler_output=buf)  # doctest: +SKIP

After that computation, ``buf`` will have the the marshalled stats data suitable
for reading with :mod:`pstats`. This feature is useful when blaze servers are
being run behind a load balancer and we do not want to search all of the servers
to find the output.

.. note::

   Because the data is serialized with :mod:`marshal` it must be read by the
   same version of python as the server. This means that a python 2 client
   cannot unmarshal the data written by a python 3 server. This is to conform
   with the file format expected by :mod:`pstats`, the standard profiling output
   inspection library.

System administrators may also configure all computations to be profiled by
default. This is useful if the client code cannot be easily changed or threading
arguments to compute is hard in an application setting. This may be set with
``profile_by_default=True`` when constructing the server.


Conclusion
==========

Because this process builds off Blaze expressions it works equally well for data
stored in any format on which Blaze is trained, including in-memory DataFrames,
SQL/Mongo databases, or even Spark clusters.


.. _Flask : http://flask.pocoo.org/docs/0.10/quickstart/#a-minimal-application
.. _Shove : https://pypi.python.org/pypi/shove/0.5.6
.. _Chest : https://github.com/mrocklin/chest
