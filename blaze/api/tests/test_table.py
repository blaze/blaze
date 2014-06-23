from blaze.api.table import Table, compute, table_repr
from blaze.data.python import Python
from blaze.compute.core import compute
from blaze.compute.python import compute
from datashape import dshape


data = (('Alice', 100),
        ('Bob', 200))

t = Table(data, columns=['name', 'amount'])

def test_resources():
    assert t.resources() == {t: t.data}


def test_compute():
    assert compute(t) == data


def test_compute():
    assert list(compute(t['amount'] + 1)) == [101, 201]


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
    assert isinstance(result, str)
    assert 'Alice' in result

    result = table_repr(t['amount'] + 1)
    print(result)
    assert '101' in result
