## Blaze: Foundations of Array Computing



## NumPy arrays and Pandas DataFrames are *foundational data structures*


## But they are restricted to memory

This is ok 95% of cases<br>
what about the other 5%?


## Computational Projects

Excellent streaming, out-of-core, and distributed alternatives exist

### NumPy like

*   SciDB
*   h5py
*   DistArray
*   Elemental
*   PETCs, Trillinos
*   Biggus
*   ...

Each approach is valid in a particular situation


## Computational Projects

Excellent streaming, out-of-core, and distributed alternatives exist

### Pandas like

*   Postgres/SQLite/MySQL/Oracle
*   PyTables, BColz
*   HDFS
    * Hadoop (Pig, Hive, ...)
    * Spark
    * Impala
*   ...

Each approach is valid in a particular situation


## Data Storage

Analagous variety of data storage techniques
</br>

- CSV - Accessible
- JSON - Pervasive, human/machine readable
- HDF5 - Efficient binary access
- BColz - Efficient columnar access
- Parquet - Efficient columnar access
- HDFS - Big!
- SQL - SQL!

</br>
Each approach is valid in a particular situation


## Spinning up a new technology is expensive


## Keeping up with a changing landscape frustrates developers


## Foundations address these challenges by being adaptable



### Blaze connects familiar interfaces to a variety of backends

Three parts

*   Abstract expression system around Tables, Arrays
*   Dispatch system from these expressions to computational backends
*   Dispatch system between data stored in different backends


Blaze looks and feels like Pandas

```Python
>>> from blaze import *
>>> iris = CSV('examples/data/iris.csv')

>>> t = Table(iris)
>>> t.head(3)
    sepal_length  sepal_width  petal_length  petal_width      species
0            5.1          3.5           1.4          0.2  Iris-setosa
1            4.9          3.0           1.4          0.2  Iris-setosa
2            4.7          3.2           1.3          0.2  Iris-setosa

>>> t.species.distinct()
           species
0      Iris-setosa
1  Iris-versicolor
2   Iris-virginica
```


Blaze operates on various systems, like SQL

```Python
>>> from blaze import *
>>> iris = SQL('sqlite:///examples/data/iris.db', 'iris')

>>> t = Table(iris)
>>> t.head(3)
    sepal_length  sepal_width  petal_length  petal_width      species
0            5.1          3.5           1.4          0.2  Iris-setosa
1            4.9          3.0           1.4          0.2  Iris-setosa
2            4.7          3.2           1.3          0.2  Iris-setosa

>>> t.species.distinct()
           species
0      Iris-setosa
1  Iris-versicolor
2   Iris-virginica
```


... and Spark

```Python
>>> import pyspark
>>> sc = pyspark.SparkContext("local", "blaze-demo")
>>> rdd = into(sc, csv)  # handle data conversion
>>> t = Table(rdd)
>>> t.head(3)
    sepal_length  sepal_width  petal_length  petal_width      species
0            5.1          3.5           1.4          0.2  Iris-setosa
1            4.9          3.0           1.4          0.2  Iris-setosa
2            4.7          3.2           1.3          0.2  Iris-setosa

>>> t.species.distinct()
           species
0      Iris-setosa
1  Iris-versicolor
2   Iris-virginica
```


### Currently supports the following

*   Python -- (through `toolz`)
*   NumPy
*   Pandas
*   SQL -- (through `sqlalchemy`)
*   HDF5 -- (through `h5py`, `pytables`)
*   MongoDB -- (through `pymongo`)
*   Spark -- (through `pyspark`)
*   Impala -- (through `impyla`, `sqlalchemy`)


Blaze organizes other open source projects to achieve a cohesive and flexible data analytics engine

</br></br>
Blaze doesn't do any real work.

It orchestrates functionality already in the Python ecosystem.

