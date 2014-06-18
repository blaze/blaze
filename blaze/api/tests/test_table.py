from blaze.api.table import Table, compute
from blaze.compute.core import compute
from blaze.compute.python import compute


data = (('Alice', 100),
        ('Bob', 200))

t = Table(data, columns=['name', 'amount'])

def test_resources():
    assert t.resources() == {t: t.data}


def test_compute():
    assert compute(t) == data

def test_compute():
    assert list(compute(t['amount'] + 1)) == [101, 201]
