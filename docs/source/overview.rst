========
Overview
========

Blaze Abstracts Computation and Storage
---------------------------------------

.. image:: svg/numpy_plus.png
    :align: center


Several projects provide rich and performant data analytics.  Competition
between these projects gives rise to a vibrant and dynamic ecosystem.
Blaze augments this ecosystem with a uniform and adaptable interface.  Blaze
orchestrates computation and data access among these external projects.  It
provides a consistent backdrop to build standard interfaces usable by the
current Python community.


Demonstration
-------------

Blaze separates the computations that we want to perform:

.. code-block:: python

   >>> from blaze import *
   >>> accounts = Symbol('accounts', 'var * {id: int, name: string, amount: int}')

   >>> deadbeats = accounts[accounts.amount < 0].name

From the representation of data

.. code-block:: python

   >>> L = [[1, 'Alice',   100],
   ...      [2, 'Bob',    -200],
   ...      [3, 'Charlie', 300],
   ...      [4, 'Denis',   400],
   ...      [5, 'Edith',  -500]]

Blaze enables users to solve data-oriented problems

.. code-block:: python

   >>> list(compute(deadbeats, L))
   ['Bob', 'Edith']

But the separation of expression from data allows us to switch between
different backends.

Here we solve the same problem using Pandas instead of Pure Python.

.. code-block:: python

   >>> df = DataFrame(L, columns=['id', 'name', 'amount'])

   >>> compute(deadbeats, df)
   1      Bob
   4    Edith
   Name: name, dtype: object

Blaze doesn't compute these results, Blaze intelligently drives other projects
to compute them instead.  These projects range from simple Pure Python
iterators to powerful distributed Spark clusters.  Blaze is built to be
extended to new systems as they evolve.

Scope
-----

Blaze speaks Python and Pandas as seen above and also several other
technologies, including NumPy, SQL, Mongo, Spark, PyTables, etc..  Blaze is
built to make connecting to a new technology easy.

Blaze currently targets database and array technologies used for analytic
queries.  It strives to orchestrate and provide interfaces on top of and in
between other computational systems.  We provide performance by providing data
scientists with intuitive access to a variety of tools.
