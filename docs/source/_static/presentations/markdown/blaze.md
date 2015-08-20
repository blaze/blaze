### `blaze` - a user interface

<img src="http://blaze.pydata.org/docs/dev/_images/xyz.png">

```python
>>> z = log(x - 1)**y
```


### We often link interface and implementation

this yields both good and bad consequences


<object data="images/frontbackends-numpy-pandas.svg"
        type="image/svg+xml"
        width="100%">
</object>


<object data="images/frontbackends-without-blaze.svg"
        type="image/svg+xml"
        width="100%">
</object>


<object data="images/frontbackends-with-blaze.svg"
        type="image/svg+xml"
        width="100%">
</object>


### Blaze is a single interface to query many systems

*demo*


Blaze separates our intent:

```python
>>> from blaze.expr import Symbol
>>> bank = Symbol('bank', 'var * {id:int, name:string, balance:int}')

>>> deadbeats = bank[bank.balance < 0].name
```

from the data:

```python
>>> L = [[1, 'Alice',   100],
...      [2, 'Bob',    -200],
...      [3, 'Charlie', 300],
...      [4, 'Dennis',  400],
...      [5, 'Edith',  -500]]
...
```

then combines the two explicitly

```python
>>> from blaze.compute import compute
>>> compute(deadbeats, L)   # Iterator in, Iterator out
<itertools.imap at 0x7fce75a9f790>
>>> list(_)
['Bob', 'Edith']
```


Separating intent from data lets us switch backends

```python
>>> from blaze.expr import Symbol
>>> bank = Symbol('bank', 'var * {id:int, name:string, balance:int}')

>>> deadbeats = bank[bank.balance < 0].name
```

so we can drive Pandas instead

```python
>>> df = DataFrame([[1, 'Alice',   100],
...                 [2, 'Bob',    -200],
...                 [3, 'Charlie', 300],
...                 [4, 'Dennis',  400],
...                 [5, 'Edith',  -500]],
...                 columns=['id', 'name', 'balance'])
```

getting the same result through different means

```python
>>> from blaze.compute import compute
>>> compute(deadbeats, df)  # DataFrame in, DataFrame out
1      Bob
4    Edith
Name: name, dtype: object
```


Now we reach out into the ecosystem

```python
>>> from blaze.expr import Symbol
>>> bank = Symbol('bank', 'var * {id:int, name:string, balance:int}')

>>> deadbeats = bank[bank.balance < 0].name
```

and use newer technologies

```python
>>> import pyspark
>>> sc = pyspark.SparkContext('local', 'Blaze-demo')

>>> rdd = into(sc, L)  # migrate to Resilient Distributed Dataset (RDD)
>>> rdd
ParallelCollectionRDD[0] at parallelize at PythonRDD.scala:315
```

evolving with the ecosystem

```python
>>> from blaze.compute import compute
>>> compute(deadbeats, rdd) # RDD in, RDD out
PythonRDD[1] at RDD at PythonRDD.scala:43
>>> _.collect()             # Pull results down to local Python
['Bob', 'Edith']
```


### Why separate expressions from computation?

1.  Write once, run anywhere
2.  Scalable development

    (start with CSV files, end with Impala/Spark)
3.  Rapid prototyping

    (try Postgres, MongoDB, Spark, see what suits you best)
4.  Robust to changes in architecture

    (assuming Blaze supports Hadoop++)
5.  Cross-backend query optimization

    [NYCTaxi CSV example](http://nbviewer.ipython.org/url/blaze.pydata.org/notebooks/timings-csv.ipynb)


### Things Blaze Can't Do

Blaze is generic (that's the point) but we give up *a lot*:

*   Blaze is not itself a database
*   Blaze is not a Pandas/Spark replacement
*   Blaze can't do things that are hard to do in parallel (e.g. median,
    full sorting, explicit groupings, quantiles)
*   Blaze can't do things that the underlying database can't do (e.g. no joins
    in Mongo)


### Questions?

* Source: [https://github.com/blaze/blaze](https://github.com/blaze/blaze)
* Docs: [http://blaze.pydata.org/](http://blaze.pydata.org/)

```python
>>> import blaze as bz
>>> iris = bz.Data('iris.csv')                     # From the small
>>> db = bz.Data('impala://54.24.132.22/default')  # To the large
...
```
