========
Examples
========

Blaze can help solve many common problems that data analysts and scientists encounter. Here are a few examples of common issues that can be solved using  blaze.

--------------------------------------
Combining separate, gzipped csv files.
--------------------------------------

.. doctest::

   >>> from blaze import *
   >>> from os import listdir
   >>> into(DataFrame, 'examples/data/accounts_*.csv')
      id     name  amount
   0   1    Alice     100
   1   2      Bob     200
   2   3  Charlie     300
   3   4      Dan     400
   4   5    Edith     500
   >>> into(DataFrame, 'examples/data/accounts_*.csv.gz')
      id     name  amount
   0   1    Alice     100
   1   2      Bob     200
   2   3  Charlie     300
   3   4      Dan     400
   4   5    Edith     500



-------------------
Split-Apply-Combine
-------------------

.. doctest::

   >>> from blaze import *
   >>> sql = SQL('sqlite:///examples/data/iris.db', 'iris')
   >>> t = Table(sql)
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
   >>> transform(t, petal_ratio=t.petal_length / t.petal_width)
       sepal_length  sepal_width  petal_length  petal_width      species  \
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
   
       petal_ratio  
   0      7.000000  
   1      7.000000  
   2      6.500000  
   3      7.500000  
   4      7.000000  
   5      4.250000  
   6      4.666667  
   7      7.500000  
   8      7.000000  
   9     15.000000  
   ...
   >>> by(t.species, max=t.petal_length.max(), min=t.petal_length.min())
              species  max  min
   0      Iris-setosa  1.9  1.0
   1  Iris-versicolor  5.1  3.0
   2   Iris-virginica  6.9  4.5


