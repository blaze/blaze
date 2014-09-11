Quickstart
===========

This quickstart is here to show some simple ways to get started created
and manipulating Blaze Tables. To run these examples, import blaze
as follows.

.. code-block:: python

    >>> from blaze import *

Blaze Tables
~~~~~~~~~~~~

Create simple Blaze tables from nested lists/tuples. Blaze will deduce the
dimensionality and data type to use.

.. code-block:: python

    >>> t = Table([(1, 'Alice', 100),
    ...            (2, 'Bob', -200),
    ...            (3, 'Charlie', 300),
    ...            (4, 'Denis', 400),
    ...            (5, 'Edith', -500)],
    ...            columns=['id', 'name', 'balance'])

    >>> t
       id     name  balance
    0   1    Alice      100
    1   2      Bob     -200
    2   3  Charlie      300
    3   4    Denis      400
    4   5    Edith     -500

    [5 rows x 3 columns]


Simple Calculations
~~~~~~~~~~~~~~~~~~~

Blaze supports simple computations like column selection and filtering
with familiar Pandas getitem or attribute syntax.

.. code-block:: python

   >>> t[t['balance'] < 0]
      id   name  balance
   0   2    Bob     -200
   1   5  Edith     -500

   >>> t[t.balance < 0]
      id   name  balance
   0   2    Bob     -200
   1   5  Edith     -500

   >>> t[t.balance < 0].name
       name
   0    Bob
   1  Edith


Stored Data
~~~~~~~~~~~

Define Blaze Tables directly from storage like CSV or HDF5 files.  Here we
operate on a CSV file of the traditional `iris dataset`_.

.. code-block:: python

   >>> iris = Table(CSV('examples/data/iris.csv'))
   >>> iris
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

Use remote data like SQL databases or Spark resilient distributed
data-structures in exactly the same way.  Here we operate on a SQL database
stored in a `sqlite file`_.

.. code-block:: python

   >>> from blaze.sql import *
   >>> sql = SQL('sqlite:///examples/data/iris.db', 'iris')
   >>> iris = Table(sql)
   >>> iris
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

More Computations
~~~~~~~~~~~~~~~~~

Common operations like Joins and split-apply-combine are available on any kind
of data

.. code-block:: python

   >>> by(iris.species,                # Group by species
   ...    min=iris.petal_width.min(),     # Minimum of the petal_width column per group
   ...    max=iris.petal_width.max())    # Maximum of the petal_width column per group
              species  max  min
   0      Iris-setosa  0.6  0.1
   1  Iris-versicolor  1.8  1.0
   2   Iris-virginica  2.5  1.4

Finishing Up
~~~~~~~~~~~~

Blaze computes only as much as is necessary to present the results on screen.
Fully evaluate the computation, returning an output similar to the input type
by calling ``compute``.

.. code-block:: python

   >>> t[t.balance < 0].name                  # Still a Table Expression
       name
   0    Bob
   1  Edith

   >>> list(compute(t[t.balance < 0].name))   # Just a raw list
   ['Bob', 'Edith']

Alternatively use the ``into`` operation to push your output into a suitable
container type.

.. code-block:: python

   >>> result = by(iris.species,
   ...             iris.petal_width.mean())

   >>> result_list = into(list, result)                     # Push result into a list

   >>> into(DataFrame, result)                # Push result into a DataFrame
              species  petal_width_mean
   0      Iris-setosa             0.246
   1  Iris-versicolor             1.326
   2   Iris-virginica             2.026

   >>> csv = CSV('examples/data/output.csv', schema=result.schema, mode='w')
   >>> write_to = into(csv, result)                      # Write result to CSV file

.. _`iris dataset`: https://raw.githubusercontent.com/ContinuumIO/blaze/master/examples/data/iris.csv
.. _`sqlite file`: https://raw.githubusercontent.com/ContinuumIO/blaze/master/examples/data/iris.db
