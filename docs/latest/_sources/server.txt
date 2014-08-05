======
Server
======

Blaze data descriptors provide uniform NumPy-like access to a variety of common
data formats.  Blaze Server builds off of this uniform interface to host data
remotely through a simple REST-ful API.

To demonstrate the use of the Blaze server we first build a data descriptor
from a CSV file.

.. code-block:: python

   >>> from blaze import *
   >>> data = CSV('accounts.csv')
   >>> data.py[0]
   (1L, 'Alice', 100L)

Then we host this data descriptor under the name ``'accounts'``.  A Server is a
mapping of names to datasets and a Flask app.

.. code-block:: python

   >>> from blaze.serve import Server
   >>> server = Server({'accounts': data})
   >>> server.app.run(host='0.0.0.0', port=5000)

With this code our machine is now hosting our CSV file through a
web-application on port 5000.  We can now access our CSV file, through Blaze,
as a service from a variety of applications.

Using the Python Requests library

.. code-block:: python

>>> import json
>>> import requests

>>> r = requests.get('http://localhost:5000/data/accounts.json',
...                 data=json.dumps({'index': 0}),
...                 headers={'Content-Type': 'application/json'})

>>> json.loads(r.content)
{u'data': [1, u'Alice', 100],
 u'datashape': u'{ id : int64, name : string, balance : int64 }',
 u'index': 0,
 u'name': u'accounts'}

Or using ``curl``

..

   $ curl -H "Content-Type: application/json" -d '{"index": 0}'
   localhost:5000/data/accounts.json
   {
     "data": [1, "Alice", 100],
     "datashape": "{ id : int64, name : string, balance : int64 }",
     "index": 0,
     "name": "accounts"
    }

Because this process builds off of the Blaze data layer it would would work
equally well for data stored in CSV files, HDF5 files, or even a SQL database.

Valid inputs for ``"index"`` include integers for row numbers, slices in the
form ``{"start": 0, "stop": 100, "step": 10}``, column names or lists of column
names like the following request which gets the first ten name, balance pairs:

..

    data='{"index": [{"start": 0, "stop": 10}, ["name", "balance"]]}'


Client
------

Just as the ``Server`` object can host a data descriptor on the network, the
``Client`` data descriptor consumes the API to provide the same data descriptor
at a remote location,  removing the need to use the API if the client
application is also in Python.

.. code-block:: python

   >>> from blaze.serve import Client
   >>> dd = Client('http://hostname:5000/', 'accounts')

   >>> dd.columns
   [u'id', u'name', u'balance']

   >>> dd.py[:3, ['name', 'balance']]
   [(u'Alice', 100L),
    (u'Bob', -200L),
    (u'Charlie', 300L)]

These actions query the server hosted at ``hostname:5000`` for the data
descriptor associated to the name accounts.
