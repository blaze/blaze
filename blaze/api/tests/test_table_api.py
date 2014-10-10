from blaze.api.table import Table, compute, concrete_head, expr_repr
from blaze.api.into import into
import os

from blaze.data.python import Python
from blaze.data import CSV
from blaze.compute.core import compute
from blaze.compute.python import compute
from blaze.expr.table import TableSymbol
from datashape import dshape
from blaze.utils import tmpfile, example
import pytest

import pandas as pd
import numpy as np
from dynd import nd

data = (('Alice', 100),
        ('Bob', 200))

L = [[1, 'Alice',   100],
     [2, 'Bob',    -200],
     [3, 'Charlie', 300],
     [4, 'Denis',   400],
     [5, 'Edith',  -500]]

t = Table(data, columns=['name', 'amount'])


def test_table_raises_on_inconsistent_inputs():
    with pytest.raises(ValueError):
        t = Table(data, schema='{name: string, amount: float32}',
            dshape=dshape("{ name : string, amount : float32 }"))


def test_table_raises_when_not_given_enough_meta_information():
    with pytest.raises(TypeError):
        t = Table([1,2,3,4,5])


def test_table_works_on_single_vectors():
    ll = Table([1,2,3,4,5], columns='numbers')
    lt = Table((1,2,3,4,5), columns='numbers')
    lnp = Table(np.array([1,2,3,4,5]), columns='numbers')


def test_resources():
    assert t.resources() == {t: t.data}


def test_resources_fail():
    t = TableSymbol('t', '{x:int, y:int}')
    d = t[t['x'] > 100]
    with pytest.raises(ValueError):
        compute(d)


def test_compute():
    assert compute(t) == data


def test_len():
    assert len(t) == 2
    assert len(t.name) == 2


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
    schema = '{name: string, amount: int64}'
    ddesc = Python(data, schema=schema)
    t = Table(ddesc)
    assert t.schema == dshape(schema)
    assert t.name
    assert t.data == ddesc


def test_repr():
    result = expr_repr(t['name'])
    print(result)
    assert isinstance(result, str)
    assert 'Alice' in result
    assert 'Bob' in result
    assert '...' not in result

    result = expr_repr(t['amount'] + 1)
    print(result)
    assert '101' in result

    t2 = Table(tuple((i, i**2) for i in range(100)), columns=['x', 'y'])
    result = expr_repr(t2)
    print(result)
    assert len(result.split('\n')) < 20
    assert '...' in result


def test_repr_of_scalar():
    assert repr(t.amount.sum()) == '300'


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


def test_expr_repr_empty():
    s = repr(t[t.amount > 1e9])
    assert isinstance(s, str)
    assert 'amount' in s


def test_to_html():
    s = t.to_html()
    assert s
    assert 'Alice' in s
    assert '<table' in s


def test_into():
    from blaze.api.into import into
    assert into([], t) == into([], data)


def test_serialization():
    import pickle
    t2 = pickle.loads(pickle.dumps(t))

    assert t.schema == t2.schema
    assert t._name == t2._name


def test_table_resource():
    with tmpfile('csv') as filename:
        csv = CSV(filename, 'w', schema='{x: int, y: int}')
        csv.extend([[1, 2], [10, 20]])

        t = Table(filename)
        assert isinstance(t.data, CSV)
        assert list(compute(t)) == list(csv)


def test_concretehead_failure():
    t = TableSymbol('t', '{x:int, y:int}')
    d = t[t['x'] > 100]
    with pytest.raises(ValueError):
        concrete_head(d)


def test_into_np_ndarray_column():
    tble = Table(L, columns=['id', 'name', 'balance'])
    expr = tble[tble['balance'] < 0]['name']
    colarray = into(np.ndarray, expr)
    assert len(list(compute(expr))) == len(colarray)


def test_into_nd_array_selection():
    tble = Table(L, columns=['id', 'name', 'balance'])
    expr = tble[tble['balance'] < 0]
    selarray = into(nd.array, expr)
    assert len(list(compute(expr))) == len(selarray)


def test_into_nd_array_column_failure():
    tble = Table(L, columns=['id', 'name', 'balance'])
    expr = tble[tble['balance'] < 0]
    colarray = into(nd.array, expr)
    assert len(list(compute(expr))) == len(colarray)


def test_table_attribute_repr():
    path = os.path.join(os.path.dirname(__file__), 'accounts.csv')
    t = Table(CSV(path))
    result = t.timestamp.day
    expected = pd.DataFrame({'timestamp_day': [25] * 3})
    assert repr(result) == repr(expected)


def test_can_trivially_create_csv_table():
    Table(example('iris.csv'))


def test_can_trivially_create_sqlite_table():
    Table('sqlite:///'+example('iris.db')+'::iris')


def test_can_trivially_create_pytables():
    Table(example('accounts.h5')+'::/accounts')
