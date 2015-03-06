========
Examples
========

Blaze can help solve many common problems that data analysts and scientists encounter. Here are a few examples of common issues that can be solved using  blaze.

--------------------------------------
Combining separate, gzipped csv files.
--------------------------------------

.. doctest::

   >>> from blaze import odo
   >>> from pandas import DataFrame
   >>> odo('blaze/examples/data/accounts_*.csv.gz', DataFrame)
      id      name  amount
   0   1     Alice     100
   1   2       Bob     200
   2   3   Charlie     300
   3   4       Dan     400
   4   5     Edith     500


-------------------
Split-Apply-Combine
-------------------

.. doctest::

   >>> from blaze import Data, by
   >>> t = Data('sqlite:///blaze/examples/data/iris.db::iris')
   >>> t
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
