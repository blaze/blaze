=============
Basic Queries
=============

Here we give a quick overview of some of the more common query functionality.

We use the well known iris dataset

.. code-block:: python

   >>> from blaze import data
   >>> from blaze.utils import example
   >>> iris = data(example('iris.csv'))
   >>> iris.peek()
       sepal_length  sepal_width  petal_length  petal_width      species
   0            5.1          3.5           1.4          0.2  Iris-setosa
   1            4.9          3.0           1.4          0.2  Iris-setosa
   2            4.7          3.2           1.3          0.2  Iris-setosa
   3            4.6          3.1           1.5          0.2  Iris-setosa
   ...


Column Access
-------------

Select individual columns using attributes

.. code-block:: python

   >>> iris.species
           species
   0   Iris-setosa
   1   Iris-setosa
   2   Iris-setosa
   3   Iris-setosa
   ...

Or item access

.. code-block:: python

   >>> iris['species']
           species
   0   Iris-setosa
   1   Iris-setosa
   2   Iris-setosa
   3   Iris-setosa
   ...

Select many columns using a list of names

.. code-block:: python

   >>> iris[['sepal_length', 'species']]
       sepal_length      species
   0            5.1  Iris-setosa
   1            4.9  Iris-setosa
   2            4.7  Iris-setosa
   3            4.6  Iris-setosa
   ...


Mathematical operations
-----------------------

Use mathematical operators and functions as normal

.. code-block:: python

   >>> from blaze import log
   >>> log(iris.sepal_length * 10)
       sepal_length
   0       3.931826
   1       3.891820
   2       3.850148
   3       3.828641
   ...


Note that mathematical functions like ``log`` should be imported from ``blaze``.
These will translate to ``np.log``, ``math.log``, ``sqlalchemy.sql.func.log``,
etc. based on the backend.


Reductions
----------

As with many Blaze operations reductions like ``sum`` and ``mean`` may be used
either as methods or as base functions.

.. code-block:: python

   >>> iris.sepal_length.mean()  # doctest: +ELLIPSIS
   5.84333333333333...

   >>> from blaze import mean
   >>> mean(iris.sepal_length)  # doctest: +ELLIPSIS
   5.84333333333333...


Split-Apply-Combine
-------------------

The ``by`` operation expresses split-apply-combine computations.  It has the
general format

.. code-block:: python

   >>> by(table.grouping_columns, name_1=table.column.reduction(),
   ...                            name_2=table.column.reduction(),
   ...                            ...)  # doctest: +SKIP

Here is a concrete example.  Find the shortest, longest, and average petal
length by species.

.. code-block:: python

   >>> from blaze import by
   >>> by(iris.species, shortest=iris.petal_length.min(),
   ...                   longest=iris.petal_length.max(),
   ...                   average=iris.petal_length.mean())
              species  average  longest  shortest
   0      Iris-setosa    1.462      1.9       1.0
   1  Iris-versicolor    4.260      5.1       3.0
   2   Iris-virginica    5.552      6.9       4.5

This simple model can be extended to include more complex groupers and more
complex reduction expressions.


Add Computed Columns
--------------------

Add new columns using the ``transform`` function

.. code-block:: python

   >>> transform(iris, sepal_ratio = iris.sepal_length / iris.sepal_width,
   ...                 petal_ratio = iris.petal_length / iris.petal_width)  # doctest: +SKIP
       sepal_length  sepal_width  petal_length  petal_width      species  \
   0            5.1          3.5           1.4          0.2  Iris-setosa
   1            4.9          3.0           1.4          0.2  Iris-setosa
   2            4.7          3.2           1.3          0.2  Iris-setosa
   3            4.6          3.1           1.5          0.2  Iris-setosa

       sepal_ratio  petal_ratio
   0      1.457143     7.000000
   1      1.633333     7.000000
   2      1.468750     6.500000
   3      1.483871     7.500000
   ...


Text Matching
-------------

Match text with glob strings, specifying columns with keyword arguments.

.. code-block:: python

   >>> iris[iris.species.like('*versicolor')]  # doctest: +SKIP
       sepal_length  sepal_width  petal_length  petal_width          species
   50           7.0          3.2           4.7          1.4  Iris-versicolor
   51           6.4          3.2           4.5          1.5  Iris-versicolor
   52           6.9          3.1           4.9          1.5  Iris-versicolor


Relabel Column names
--------------------

.. code-block:: python

   >>> iris.relabel(petal_length='PETAL-LENGTH', petal_width='PETAL-WIDTH')  # doctest: +SKIP
       sepal_length  sepal_width  PETAL-LENGTH  PETAL-WIDTH      species
   0            5.1          3.5           1.4          0.2  Iris-setosa
   1            4.9          3.0           1.4          0.2  Iris-setosa
   2            4.7          3.2           1.3          0.2  Iris-setosa

========
Examples
========

Blaze can help solve many common problems that data analysts and scientists encounter. Here are a few examples of common issues that can be solved using  blaze.

Combining separate, gzipped csv files.
--------------------------------------

.. code-block:: python

   >>> from blaze import odo
   >>> from pandas import DataFrame
   >>> odo(example('accounts_*.csv.gz'), DataFrame)
      id      name  amount
   0   1     Alice     100
   1   2       Bob     200
   2   3   Charlie     300
   3   4       Dan     400
   4   5     Edith     500


Split-Apply-Combine
-------------------

.. code-block:: python

   >>> from blaze import data, by
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
   >>> by(t.species, max=t.petal_length.max(), min=t.petal_length.min())
              species  max  min
   0      Iris-setosa  1.9  1.0
   1  Iris-versicolor  5.1  3.0
   2   Iris-virginica  6.9  4.5
