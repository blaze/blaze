Quickstart
===========

This quickstart is here to show some simple ways to get started created
and manipulating Blaze Tables. To run these examples, import blaze
as follows.

.. doctest::

    >>> from blaze import *

Blaze Arrays
~~~~~~~~~~~~

To create simple Blaze arrays, you can construct them from
nested lists. Blaze will deduce the dimensionality and
data type to use.

.. doctest::

    >>> t = Table(((1, 'Alice', 100),
                   (2, 'Bob', -200),
                   (3, 'Charlie', 300),
                   (4, 'Denis', 400),
                   (5, 'Edith', -500)),
                   columns=['id', 'name', 'balance'])
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

Blaze supports column selection and filtering similarly to Pandas.

.. doctest::

   >>> t[t['balance'] < 0]
      id   name  balance
   0   2    Bob     -200
   1   5  Edith     -500

   [2 rows x 1 columns]

   >>> t[t['balance'] < 0]['name']
       name
   0    Bob
   1  Edith

   [2 rows x 1 columns]


Stored Data
~~~~~~~~~~~

Blaze Tables can also be defined directly from storage like CSV or HDF5 files.

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

This extends out to data in SQL databases or Spark resilient distributed
datastructures.

.. doctest::

   >>> from blaze.sql import *
   >>> bitcoin = Table(SQL('sqlite:///bitcoin.db', 'transactions'))
   >>> bitcoin
       transaction   sender  recipient                date       value
   0             1        2          2 2013-04-10 14:22:50   24.375000
   1             1        2     782477 2013-04-10 14:22:50    0.770900
   2             2   620423    4571210 2011-12-27 11:43:12  614.174951
   3             2   620423          3 2011-12-27 11:43:12  128.040520
   4             3        3     782479 2013-04-10 14:22:50   47.140520
   5             3        3          4 2013-04-10 14:22:50  150.000000
   6             4    39337      39337 2012-06-17 12:02:02    0.310818
   7             4    39337          3 2012-06-17 12:02:02   69.100000
   8             5  2071196    2070358 2013-03-04 14:38:05   61.602352
   9             5  2071196          5 2013-03-04 14:38:05  100.000000
   10            6        5     782480 2013-04-10 14:22:50   65.450000

   ...

In each of these cases Blaze consumes only as much as it needs to present what
is on screen.  To fully evalute the result push your computation into a
container.

More Computations
~~~~~~~~~~~~~~~~~

Common operations like Joins and split-apply-combine are available on any kind
of data

.. doctest::

   >>> By(iris,
   ...    iris['species'],
   ...    iris['petal_width'].mean())
              species  petal_width
   0   Iris-virginica        2.026
   1      Iris-setosa        0.246
   2  Iris-versicolor        1.326


   >>> By(bitcoin,
   ...    bitcoin['sender'],
   ...    bitcoin['value'].sum()).sort('value', ascending=False)
       sender            value
   0       11  52461821.941658
   1     1374  23394277.034152
   2       25  13178095.975724
   3       29   5330179.983047
   4    12564   3669712.399825
   5   782688   2929023.064648
   6       74   2122710.961163
   7    91638   2094827.825161
   8       27   2058124.131470
   9       20   1182868.148780
   10     628    977311.388250


Finishing Up
~~~~~~~~~~~~

Fully evaluate the computation, returning an output similar to the input type
by calling ``compute``.

.. doctest::

   >>> t[t['balance'] < 0]['name']              # Still a Table Expressions
       name
   0    Bob
   1  Edith

   >>> compute(t[t['balance'] < 0]['name'])     # Just a raw list
   ['Bob', 'Edith']

Alternatively use the ``into`` operation to transform your output into various
forms.

.. doctest::

   >>> result = By(iris,
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
