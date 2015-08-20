## `into(target, source)`

![](images/into-small.png)


## Q: How do you migrate a CSV file into a Mongo Database?


## Q: How do you migrate a CSV file into a Mongo Database?

    CSV -> DataFrames:                  pd.read_csv()
    DataFrames -> NumPy Arrays:         DataFrame.to_records()
    NumPy Arrays -> Iterator:           ndarray.tolist()
    Iterator -> pymongo.Collection:     Collection.insert


## Q: How do you migrate a CSV file into a Mongo Database?

```python
>>> #    target                                  source
>>> into('mongodb://localhost/db::mycollection', 'myfile.csv')
```


## Q: How do you Load a JSON file on S3 into Postgres?


## Q: How do you Load a JSON file on S3 into Postgres?

    JSON on S3 -> Local JSON:           boto
    JSON to Python iterator:            json library
    Python iterator to DataFrames:      partition_all() and DataFrame()
    DataFrames -> CSV files:            DataFrame.to_csv()
    CSV -> Postgres:                    LOAD command in Postgres


## Q: How do you Load a JSON file on S3 into Postgres?

```python
>>> #    target                                  source
>>> into('postgresql://postgres:postgres@localhost::mytable',
...      's3://mybucket/myfile.json')
```


## Data Science is hard

*  Each step is straightforward
*  The entire process is hell


### Into embraces the complexity

![](images/into-small.png)

*  Nodes are data types (`DataFrame`, `list`, `sqlalchemy.Table`, ...)
*  Edges are functions (`DataFrame -> CSV via read_csv`, ...)
*  Edges are weighted by speed, we search for the minimum path.
*  Red nodes can be larger than memory.  Transfers between two red nodes only
   use the red subgraph


### Today's graph

![](images/into-big.png)


### How to get and use `into`

    conda install into
    or
    pip install into

```python
>>> from into import into
>>> into(target, source)
```
    or
    $ into source target

*  Inputs can be
    * types -- `list` -- Create new target
    * objects -- `[1, 2, 3]` -- Append to target
    * strings -- `'myfile.csv'` -- Use regex magic


### How to extend `into`

```python
from into import convert, resource

@convert.register(np.ndarray, pd.DataFrame, cost=1.0)
def dataframe_to_numpy(df, **kwargs):
    return df.to_records(index=False)

@convert.register(list, np.ndarray, cost=10.0)
def numpy_to_list(x, **kwargs):
    return x.tolist()
```


### Questions?

* Source: [https://github.com/blaze/into](https://github.com/blaze/into)
* Docs: [http://into.readthedocs.org/en/latest/](http://into.readthedocs.org/en/latest/)
* Blog: [http://matthewrocklin.com/blog](http://matthewrocklin.com/blog)

```python
>>> from into import into
>>> happiness = into(target, source)
```
