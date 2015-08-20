
## Frames

Think about common operations on DataFrames.

What do they look like?


### Anatomy of a Dask.Frame

* Logically dask Arrays are a grid of NumPy Arrays
* Dask Frame is a sequence of Pandas DataFrames

<div align="center">
<table >
<th>
<td><b>dask.array</b></td>
<td><b>Naive dask.frame</b></td>
</th>
<tr>
<td></td>
<td><img src="images/array.png"></td>
<td><img src="images/naive-frame.png"></td>
</tr>
</table>
</div>

For arrays blockshape information is critical for algorithms

Informs which blocks communicate with which others.


### This supports the following operations

*  Elementwise operations

        df.a + df.b

*  Row-wise filtering

        df[df.a > 0]

*  Reductions

        df.a.mean()

*  Some split-apply-combine operations

        df.groupby(...).agg(...)

The Blaze chunking/streaming backend does this

People like this, but want more.


### Does not support the following operations

*  Joins

        join(a, b, 'a_column', 'b_column')

*  Split-apply-combine with more complex `transform` or `apply` combine steps

        df.groupby(...).apply(arbitrary_function)

*  Sliding window or resampling operations

        df.rolling_mean(...)

*  Anything involving multiple datasets

        A.x[B.y > 0]


### Partition on the Index values

Instead of partitioning based on the size of blocks we instead partition on
value ranges of the index.

<div align="center">
<table>
<th>
<td><b>Partition on block size</b></td>
<td><b>Partition on index value</b></td>
</th>
<tr>
<td></td>
<td><img src="images/naive-frame.png"></td>
<td><img src="images/frame.png"></td>
</tr>
</table>
</div>

Information about value ranges helps us to create dask graphs for more complex
operations (joins, sliding windows, ...)


## Lets look at pictures again...


### Reading files

    >>> import bcolz
    >>> trip = bcolz.ctable('trip.bcolz')

    >>> import dask.frame as dfr
    >>> f = dfr.from_array(trip, chunksize=20000000)

![](images/dask.from_array.png)


### Reading files

    >>> import dask.frame as dfr
    >>> f = dfr.read_csv('trip_data_1.csv', chunksize=1000000)

<img src="images/dask.read_csv.png"
    width="20%">


### Frame operations are different.  Often messier


### DataFrame

    >>> import pandas as pd
    >>> f = pd.read_csv('accounts.csv', sep=',')

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th> balance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0 </th>
      <td>   Alice</td>
      <td>  100</td>
    </tr>
    <tr>
      <th>1 </th>
      <td>     Bob</td>
      <td>  200</td>
    </tr>
    <tr>
      <th>2 </th>
      <td>   Alice</td>
      <td>  300</td>
    </tr>
    <tr>
      <th>3 </th>
      <td>   Frank</td>
      <td>  400</td>
    </tr>
    <tr>
      <th>4 </th>
      <td>     Dan</td>
      <td>  500</td>
    </tr>
    <tr>
      <th>5 </th>
      <td>   Alice</td>
      <td>  600</td>
    </tr>
    <tr>
      <th>6 </th>
      <td>   Alice</td>
      <td>  700</td>
    </tr>
    <tr>
      <th>7 </th>
      <td> Charlie</td>
      <td>  800</td>
    </tr>
    <tr>
      <th>8 </th>
      <td>   Alice</td>
      <td>  900</td>
    </tr>
    <tr>
      <th>9 </th>
      <td>   Edith</td>
      <td> 1000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>   Frank</td>
      <td> 1100</td>
    </tr>
    <tr>
      <th>11</th>
      <td>     Bob</td>
      <td> 1200</td>
    </tr>
  </tbody>
</table>


### Dask.Frame

    >>> import dask.frame as dfr
    >>> f = dfr.read_csv('accounts.csv', sep=',', chunksize=4)

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th> balance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0 </th>
      <td>   Alice</td>
      <td>  100</td>
    </tr>
    <tr>
      <th>1 </th>
      <td>     Bob</td>
      <td>  200</td>
    </tr>
    <tr>
      <th>2 </th>
      <td>   Alice</td>
      <td>  300</td>
    </tr>
    <tr>
      <th>3 </th>
      <td>   Frank</td>
      <td>  400</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th> balance</th>
    </tr>
  </thead>
    <tr>
      <th>4 </th>
      <td>     Dan</td>
      <td>  500</td>
    </tr>
    <tr>
      <th>5 </th>
      <td>   Alice</td>
      <td>  600</td>
    </tr>
    <tr>
      <th>6 </th>
      <td>   Alice</td>
      <td>  700</td>
    </tr>
    <tr>
      <th>7 </th>
      <td> Charlie</td>
      <td>  800</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th> balance</th>
    </tr>
  </thead>
    <tr>
      <th>8 </th>
      <td>   Alice</td>
      <td>  900</td>
    </tr>
    <tr>
      <th>9 </th>
      <td>   Edith</td>
      <td> 1000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>   Frank</td>
      <td> 1100</td>
    </tr>
    <tr>
      <th>11</th>
      <td>     Bob</td>
      <td> 1200</td>
    </tr>
  </tbody>
</table>


### Many Operations are the same

    >>> f.balance.sum()

![](images/dask.frame-sum.png)


### Even some complex ones

    >>> f.groupby('name').balance.sum().compute()
    name
    Alice      2600
    Bob        1400
    Charlie     800
    Dan         500
    Edith      1000
    Frank      1500
    Name: balance, dtype: int64

![](images/dask.split-apply-aggregate.png)


### But only in certain cases

*  `df.groupby(...).aggregate(...)`

    Works well for typical aggregations

    This is because we know how to break apart operations like `count` into
    `count` and `sum`

*  `df.groupby(...).apply(arbitrary_function)`

    Much harder.  We need to assemble groups together (e.g. all of the Alice's)

The Blaze chunking backend can do split-apply-aggregate well.

It will never be able to do the general apply.


### Even though these are *spelled similarly*:

*  `df.groupby(...).aggregate(...)`
*  `df.groupby(...).apply(arbitrary_function)`

### they are *computationally different*

Many operations require us to reshuffle our data.  This breaks the task
scheduling model.



The Shuffle
-----------


### Index by Name

To run arbitrary `groupby(...).apply(func)` operations we need to collect data
in to groups.

        name, balance                       name, balance
        Alice, 100                          Alice, 100
        Bob, 200                            Alice, 300
        Alice, 300                          Alice, 600
        Frank, 400                          Alice, 700
                                            Alice, 900
        name, balance
        Dan, 500                            name, balance
        Alice, 600       -> Shuffle ->      Bob, 200
        Alice, 700                          Dan, 500
        Charlie, 800                        Bob, 1200
                                            Charlie, 800
        name, balance
        Alice, 900                          name, balance
        Edith, 1000                         Frank, 400
        Frank, 1100                         Edith, 1000
        Bob, 1200                           Frank, 1100

1.  Find values on which to partition

        (-oo, Bob), [Bob, Edith), [Edith, oo)

2.  Shard, communicate, concatenate


### Find Good Partitions - By Approximate Quantiles

Now we find approximate quantiles.  To find 100 evenly spaced groups:

1.  Call the following on each block

        np.percentile(df['new-index-column'], range(100))

2.  Collect and merge these results together intelligently (thanks Erik!)

This gets us the right values on which to shard our data

    Bob, Edith -> (-oo, Bob), [Bob, Edith), [Edith, oo)



### Find Good Partitions - By Out-of-Core Sorting

We used to perform an external sort.  This was kinda slow but could be improved.

We might want to try this again, but with more Cython.


### Shard

Split old blocks, dump shards to dict

        name, balance                       name, balance
        Alice, 100                          Alice, 100
        Bob, 200           -> Shard ->      Alice, 300       -> dict
        Alice, 300
        Frank, 400                          name, balance
                                            Bob, 200         -> dict

                                            name, balance
                                            Frank, 400       -> dict

        name, balance                       name, balance
        Dan, 500                            Alice, 600
        Alice, 600         -> Shard ->      Alice, 700       -> dict
        Alice, 700
        Charlie, 800                        name, balance
                                            Dan, 500         -> dict
                                            Charlie, 800
                               ...


### Collect

Pull shards from dict, construct new blocks

                  name, balance                     name, balance
                  Alice, 100                        Alice, 100
        dict ->   Alice, 300                        Alice, 300
                                   -> collect ->    Alice, 600
                  name, balance                     Alice, 700
        dict ->   Alice, 600
                  Alice, 700

                  name, balance
        dict ->   Bob, 200                          name, balance
                                   -> collect ->    Bob, 200
                  name, balance                     Dan, 500
        dict ->   Dan, 500                          Charlie, 800
                  Charlie, 800
                                        ...


### `dict < MutableMapping`

The actual shuffle happens in a dict / MutableMapping

* `dict` - good for in-memory workflows
* `chest` - spills to disk
* Peer-to-peer key-value store - a fun project for the future?

<hr>

This data structure determines our shuffle capabilities



### Recent work

* BColz is sometimes slow
* Writing many small files to disk is a great way to crush a computer
* Serialization costs vary (`msgpack` oddly fast?)
* Serialization of object arrays is going to be a pain

    (maybe push on categoricals?)


### Split Financial data by stock

    import dask.frame as dfr
    df = dfr.read_csv('20140616-r-00032', sep='\t',
                      names=fieldnames,
                      parse_dates={'datetime': ['System Date', 'System Time']},
                      usecols=['System Date', 'System Time', 'Symbol'])

    # Grab list of unique symbols
    symbols = list(df.Symbol.drop_duplicates().compute().sort())

    # Shard and write to disk
    def write_file(df):
        df.to_csv('stocks/' + df.index[0] + '.csv')
    df2 = df.set_partition('Symbol', symbols)
    df2.map_blocks(write_file).compute()

    mrocklin@workstation:~/data/xdata/stocks$ ls
    cAUD.CAD,(non_opt).csv  fNG.H15,(non_opt).csv    zBZ.V14_X14,(non_opt).csv
    cAUD.CHF,(non_opt).csv  fNG.J15,(non_opt).csv    zBZ.V14_Z14,(non_opt).csv
    cAUD.JPY,(non_opt).csv  fNG.K15,(non_opt).csv    zBZ.X14_F15,(non_opt).csv
    cAUD.NZD,(non_opt).csv  fNG.M15,(non_opt).csv    zBZ.X14_Z14,(non_opt).csv
    cAUD.USD,(non_opt).csv  fNG.N14,(non_opt).csv    zBZ.Z14_F15,(non_opt).csv
    cCAD.CHF,(non_opt).csv  fNG.N15,(non_opt).csv    zBZ.Z14_G15,(non_opt).csv
    ...


### Work to do

* Near term
    *  Still banging away on Shuffle
    *  A few interesting operations join, sliding window
    *  Easy support for categories

        (probably essential for performance on text)

    *  There is a lot of Pandas API

* Bigger thoughts
    *  GIL
    *  HDFS aware scheduler
    *  Peer-to-peer distributed dict


Questions?
----------

        name, balance                       name, balance
        Alice, 100                          Alice, 100
        Bob, 200                            Alice, 300
        Alice, 300                          Alice, 600
        Frank, 400                          Alice, 700
                                            Alice, 900
        name, balance
        Dan, 500                            name, balance
        Alice, 600       -> Shuffle ->      Bob, 200
        Alice, 700                          Dan, 500
        Charlie, 800                        Bob, 1200
                                            Charlie, 800
        name, balance
        Alice, 900                          name, balance
        Edith, 1000                         Frank, 400
        Frank, 1100                         Edith, 1000
        Bob, 1200                           Frank, 1100

