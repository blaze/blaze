## How does Blaze work?

At its core, Blaze is the following:

1.  Symbolic expression system -- Mathematica for data
2.  Interpreters to various backends
3.  User interface to make expression system accessible
4.  Dispatch system to make interpreters feasible

<br>

In practice, connecting to a new backend takes days, not months.


Blaze separates our intent:

```python
>>> from blaze.expr import TableSymbol
>>> bank = TableSymbol('bank', '{id:int, name:string, balance:int}')

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
>>> from blaze.expr import TableSymbol
>>> bank = TableSymbol('bank', '{id:int, name:string, balance:int}')

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
>>> from blaze.expr import TableSymbol
>>> bank = TableSymbol('bank', '{id:int, name:string, balance:int}')

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
4.  Rapid prototyping and system discovery

    (try Postgres, MongoDB, Spark, see what suits you best)
3.  Robust to changes in architecture

    (assuming Blaze will support Hadoop++)
