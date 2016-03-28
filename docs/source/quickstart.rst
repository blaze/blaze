Quickstart
===========

This quickstart is here to show some simple ways to get started created
and manipulating Blaze Symbols. To run these examples, import blaze
as follows.

.. code-block:: python

    >>> from blaze import *

Blaze Interactive Data
~~~~~~~~~~~~~~~~~~~~~~

Create simple Blaze expressions from nested lists/tuples. Blaze will deduce the
dimensionality and data type to use.

.. code-block:: python

    >>> t = data([(1, 'Alice', 100),
    ...           (2, 'Bob', -200),
    ...           (3, 'Charlie', 300),
    ...           (4, 'Denis', 400),
    ...           (5, 'Edith', -500)],
    ...          fields=['id', 'name', 'balance'])

    >>> t.peek()
       id     name  balance
    0   1    Alice      100
    1   2      Bob     -200
    2   3  Charlie      300
    3   4    Denis      400
    4   5    Edith     -500


Simple Calculations
~~~~~~~~~~~~~~~~~~~

Blaze supports simple computations like column selection and filtering
with familiar Pandas getitem or attribute syntax.

.. code-block:: python

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

Define Blaze expressions directly from storage like CSV or HDF5 files.  Here we
operate on a CSV file of the traditional `iris dataset`_.

.. code-block:: python

   >>> from blaze.utils import example
   >>> iris = data(example('iris.csv'))
   >>> iris.peek()
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

   >>> iris = data('sqlite:///%s::iris' % example('iris.db'))
   >>> iris.peek()
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
   ...    min=iris.petal_width.min(),  # Minimum of petal_width per group
   ...    max=iris.petal_width.max())  # Maximum of petal_width per group
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

   >>> t[t.balance < 0].name                  # Still an Expression
       name
   0    Bob
   1  Edith

   >>> list(compute(t[t.balance < 0].name))   # Just a raw list
   ['Bob', 'Edith']

Alternatively use the ``odo`` operation to push your output into a suitable
container type.

.. code-block:: python

   >>> result = by(iris.species, avg=iris.petal_width.mean())
   >>> result_list = odo(result, list)  # Push result into a list
   >>> odo(result, DataFrame)  # Push result into a DataFrame
              species    avg
   0      Iris-setosa  0.246
   1  Iris-versicolor  1.326
   2   Iris-virginica  2.026
   >>> odo(result, example('output.csv'))  # Write result to CSV file
   <odo.backends.csv.CSV object at ...>


.. _`iris dataset`: https://raw.githubusercontent.com/blaze/blaze/master/blaze/examples/data/iris.csv
.. _`sqlite file`: https://raw.githubusercontent.com/blaze/blaze/master/blaze/examples/data/iris.db
