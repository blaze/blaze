Quickstart
===========

This quickstart is here to show some simple ways to get started created
and manipulating Blaze Tables. To run these examples, import blaze
as follows.

.. doctest::

    >>> from blaze import *

Blaze Tables
~~~~~~~~~~~~

Create simple Blaze tables from nested lists/tuples. Blaze will deduce the
dimensionality and data type to use.

.. doctest::

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
with familiar Pandas getitem syntax.

.. doctest::

   >>> t[t['balance'] < 0]
      id   name  balance
   0   2    Bob     -200
   1   5  Edith     -500

   [2 rows x 3 columns]

   >>> t[t['balance'] < 0]['name']
       name
   0    Bob
   1  Edith

   [2 rows x 1 columns]


Stored Data
~~~~~~~~~~~

Define Blaze Tables directly from storage like CSV or HDF5 files.  Here we
operate on a CSV file of the traditional `iris dataset`_.

.. doctest::

   >>> iris = Table(CSV('iris.csv'))
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
   10           5.4          3.7           1.5          0.2  Iris-setosa

   ...

Use remote data like SQL databases or Spark resilient distributed
data-structures in exactly the same way.  Here we operate on a SQL database
stored in a `sqlite file`_.

.. doctest::

   >>> from blaze.sql import *
   >>> iris = Table(SQL('sqlite:///iris.db', 'iris'))
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
   10           5.4          3.7           1.5          0.2  Iris-setosa

   ...

More Computations
~~~~~~~~~~~~~~~~~

Common operations like Joins and split-apply-combine are available on any kind
of data

.. doctest::

   >>> by(iris,                           # Split apply combine operation
   ...    iris['species'],                # Group by species
   ...    iris['petal_width'].mean())     # Take the mean of the petal_width column
              species  petal_width
   0   Iris-virginica        2.026
   1      Iris-setosa        0.246
   2  Iris-versicolor        1.326


Finishing Up
~~~~~~~~~~~~

Blaze computes only as much as is necessary to present the results on screen.
Fully evaluate the computation, returning an output similar to the input type
by calling ``compute``.

.. doctest::

   >>> t[t['balance'] < 0]['name']                  # Still a Table Expression
       name
   0    Bob
   1  Edith

   >>> list(compute(t[t['balance'] < 0]['name']))   # Just a raw list
   ['Bob', 'Edith']

Alternatively use the ``into`` operation to push your output into a suitable
container type.

.. doctest::

   >>> result = by(iris,
   ...             iris['species'],
   ...             iris['petal_width'].mean())

   >>> into([], result)                       # Push result into a list
   [(u'Iris-virginica', 2.026),
    (u'Iris-setosa', 0.2459999999999999),
    (u'Iris-versicolor', 1.3259999999999998)]

   >>> from pandas import DataFrame
   >>> into(DataFrame(), result)              # Push result into a DataFrame
              species  petal_width
   0   Iris-virginica        2.026
   1      Iris-setosa        0.246
   2  Iris-versicolor        1.326

   >>> into(CSV('output.csv', schema=result.schema), # Write result to CSV file
   ...      result)

.. _`iris dataset`: https://raw.githubusercontent.com/ContinuumIO/blaze/master/examples/data/iris.csv
.. _`sqlite file`: https://raw.githubusercontent.com/ContinuumIO/blaze/master/examples/data/iris.db
