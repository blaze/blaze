from __future__ import absolute_import, division, print_function

from blaze.expr.table import *
from toolz import map
import math
from collections import Iterator
from blaze.compute.chunks import *
from blaze.compute.core import compute
from blaze.compute.python import *


data = [[1, 'Alice', 100],
        [2, 'Bob', 200],
        [3, 'Alice', -300],
        [4, 'Charlie', 400],
        [5, 'Edith', 200]]

t = TableSymbol('t', schema='{id: int, name: string, amount: int}')

c = Chunks(data, chunksize=2)


def test_basic():
    assert list(concat(compute(t, c))) == data


def test_column():
    assert list(concat(compute(t.id, c))) == [1, 2, 3, 4, 5]


def test_projection():
    assert list(concat(compute(t[['id', 'amount']], c))) == \
            [(1, 100), (2, 200), (3, -300), (4, 400), (5, 200)]


def test_reductions():
    assert compute(t.id.min(), c) == 1
    assert compute(t.id.max(), c) == 5
    assert compute(t.id.sum(), c) == 15
    assert compute(t.id.mean(), c) == 3
    assert compute(t.id.mean(), c) == 3


def test_distinct():
    assert sorted(compute(t.name.distinct(), c)) == \
            ['Alice', 'Bob', 'Charlie', 'Edith']

def test_nunique():
    assert compute(t.name.nunique(), c) == 4
    assert compute(t.nunique(), c) == 5


def test_columnwise():
    assert list(concat(compute(t.id + 1, c))) == [2, 3, 4, 5, 6]


def test_map():
    assert list(concat(compute(t.id.map(lambda x: x+1), c))) == [2, 3, 4, 5, 6]


def test_selection():
    assert list(concat(compute(t[t.name == 'Alice'], c))) == \
            [[1, 'Alice', 100], [3, 'Alice', -300]]


def test_compound():
    assert compute(t[t.name == 'Alice'].amount.sum(), c) == -200


def test_head():
    assert list(compute(t.head(2), c)) == list(data[:2])
    assert list(compute(t.head(3), c)) == list(data[:3])


def test_join():
    cities = TableSymbol('cities', schema='{id: int, city: string}')
    j = join(t, cities, 'id')

    city_data = [[1, 'NYC'], [1, 'Chicago'], [5, 'Paris']]

    assert set(concat(compute(join(cities, t, 'id')[['name', 'city']],
                              {t: c, cities: city_data}))) == \
            set((('Alice', 'NYC'), ('Alice', 'Chicago'), ('Edith', 'Paris')))

    assert set(concat(compute(join(t, cities, 'id')[['name', 'city']],
                              {t: c, cities: city_data}))) == \
            set((('Alice', 'NYC'), ('Alice', 'Chicago'), ('Edith', 'Paris')))


def test_by():
    assert set(compute(by(t, t.name, t.amount.sum()), c)) == \
            set([('Alice', -200), ('Bob', 200),
                 ('Charlie', 400), ('Edith', 200)])
    assert set(compute(by(t, t.name, t.amount.count()), c)) == \
            set([('Alice', 2), ('Bob', 1),
                 ('Charlie', 1), ('Edith', 1)])


def test_into_list_chunks():
    assert into([], Chunks([1, 2, 3, 4], chunksize=2)) == [1, 2, 3, 4]


def test_into_DataFrame_chunks():
    data = [['Alice', 1], ['Bob', 2], ['Charlie', 3]]
    assert str(into(DataFrame,
                    Chunks(data, chunksize=2),
                    columns=['name', 'id'])) == \
                str(DataFrame(data, columns=['name', 'id']))
