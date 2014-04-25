from blaze.compute.python import *
from blaze.objects.table import *

t = Table('{name: string, amount: int, id: int}')


data = [['Alice', 100, 1],
        ['Bob', 200, 2],
        ['Alice', 50, 3]]


def test_table():
    assert compute(t, data) == data


def test_projection():
    assert list(compute(t['name'], data)) == [x[0] for x in data]


def test_eq():
    assert list(compute(t['amount'] == 100, data)) == [x[1] == 100 for x in data]


def test_selection():
    assert list(compute(t[t['amount'] == 0], data)) == \
                [x for x in data if x[1] == 0]
    assert list(compute(t[t['amount'] > 150], data)) == \
                [x for x in data if x[1] > 150]
