## Blaze functions

Blaze uses a handful of functions:

1.  `discover(data)` -- Get metadata
2.  `compute(expr, data)` -- Execute `expr` on `data`
3.  `into(type, data)` -- Migrate `data` to new container
4.  `resource(uri)` -- Get the data behind uri string
5.  ... `drop`, `create_index`, `chunks`, ...


### We implement these functions for many different types/backends

### `discover`

Discover metadata.  <br>
Returns datashape, Blaze's internal data type system.

```python
>>> from datashape import discover

>>> discover(3.14)
dshape("float64")

>>> discover([1, 2, 3])
dshape("3 * int64")

>>> df = pd.read_csv('iris.csv')
>>> discover(df)
dshape("150 * { sepal_length : float64, sepal_width : float64,
                petal_length : float64, petal_width : float64,
                species : string }")

>>> discover(...)
```


### `compute`

Execute expression against data

```python
>>> from blaze.expr import TableSymbol
>>> bank = TableSymbol('bank', '{id:int, name:string, balance:int}')

>>> deadbeats = bank[bank.balance < 0].name

>>> L = [[1, 'Alice',   100],
...      [2, 'Bob',    -200],
...      [3, 'Charlie', 300],
...      [4, 'Dennis',  400],
...      [5, 'Edith',  -500]]

>>> from blaze.compute import compute
>>> compute(deadbeats, L)   # Iterator in, Iterator out
<itertools.imap at 0x7fab104693d0>

>>> list(_)
['Bob', 'Edith']
```


### `into`

migrate data between containers

```python
>>> into(set, [1, 2, 3])
{1, 2, 3}

>>> into(np.ndarray, df)
rec.array([(5.1, 3.5, 1.4, 0.2, 'Iris-setosa'),
           (4.9, 3.0, 1.4, 0.2, 'Iris-setosa'),
           (4.7, 3.2, 1.3, 0.2, 'Iris-setosa'),
           (4.6, 3.1, 1.5, 0.2, 'Iris-setosa'),
           ...
           (5.9, 3.0, 5.1, 1.8, 'Iris-virginica')],
     dtype=[('sepal_length', '<f8'), ('sepal_width', '<f8'),
            ('petal_length', '<f8'), ('petal_width', '<f8'),
            ('species', 'O')])

>>> db = pymongo.MongoClient().db
>>> into(db.mycollection, df)
Collection(Database(MongoClient('localhost', 27017), u'db'), u'mycoll')
```


### `resource`

find data from uri

```python
>>> resource('iris.csv')
<blaze.data.csv.CSV at 0x7fdca8f93d10>

>>> resource('sqlite:///iris.db::iris')
<blaze.data.sql.SQL at 0x7fdca8f22910>

>>> resource('mongodb://localhost:27017/db::mycoll')
Collection(Database(MongoClient('localhost', 27017), u'db'), u'mycoll')

>>> resource('accounts.h5::/accounts')
/accounts (Table(5,)) ''
  description := {
   "id": Int64Col(shape=(), dflt=0, pos=0),
   "name": StringCol(itemsize=7, shape=(), dflt='', pos=1),
   "balance": Int64Col(shape=(), dflt=0, pos=2)}
   byteorder := 'little'
   chunkshape := (2849,)
```


## Extending Blaze

You can extend these functions from outside of the Blaze codebase
(you don't need our permission)

```python
from blaze import dispatch, resource

@dispatch(MyType)
def discover(obj):
    return datashape of obj

@dispatch(blaze.expr.Head, MyType)
def compute_up(expr, myobj):
    return myobj[expr.n]

@dispatch(list, MyType)
def into(_, myobj):
    return myobj.to_list()

@resource.register(regex)
def resource(uri):
    return MyType(information-gathered-from-uri)
```
