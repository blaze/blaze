from blaze.api.table import Table, compute, table_repr
from blaze.data.python import Python
from blaze.compute.core import compute
from blaze.compute.python import compute
from datashape import dshape

import pandas as pd

data = (('Alice', 100),
        ('Bob', 200))

t = Table(data, columns=['name', 'amount'])

def test_resources():
    assert t.resources() == {t: t.data}


def test_compute():
    assert compute(t) == data


def test_compute():
    assert list(compute(t['amount'] + 1)) == [101, 201]


def test_create_with_schema():
    t = Table(data, schema='{name: string, amount: float32}')
    assert t.schema == dshape('{name: string, amount: float32}')



def test_create_with_raw_data():
    t = Table(data, columns=['name', 'amount'])
    assert t.schema == dshape('{name: string, amount: int64}')
    assert t.name
    assert t.data == data


def test_create_with_data_descriptor():
    schema='{name: string, amount: int64}'
    ddesc = Python(data, schema=schema)
    t = Table(ddesc)
    assert t.schema == dshape(schema)
    assert t.name
    assert t.data == ddesc


def test_repr():
    result = table_repr(t['name'])
    print(result)
    assert isinstance(result, str)
    assert 'Alice' in result
    assert 'Bob' in result
    assert '...' not in result

    result = table_repr(t['amount'] + 1)
    print(result)
    assert '101' in result

    t2 = Table(tuple((i, i**2) for i in range(100)), columns=['x', 'y'])
    result = table_repr(t2)
    print(result)
    assert len(result.split('\n')) < 20
    assert '...' in result


def test_mutable_backed_repr():
    mutable_backed_table = Table([[0]], columns=['col1'])
    repr(mutable_backed_table)


def test_dataframe_backed_repr():
    df = pd.DataFrame(data=[0], columns=['col1'])
    dataframe_backed_table = Table(df)
    repr(dataframe_backed_table)


def test_dataframe_backed_repr_complex():
    df = pd.DataFrame([(1, 'Alice', 100),
                       (2, 'Bob', -200),
                       (3, 'Charlie', 300),
                       (4, 'Denis', 400),
                       (5, 'Edith', -500)],
                      columns=['id', 'name', 'balance'])
    t = Table(df)
    repr(t[t['balance'] < 0])
