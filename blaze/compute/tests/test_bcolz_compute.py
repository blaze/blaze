from __future__ import absolute_import, division, print_function

import pytest
bcolz = pytest.importorskip('bcolz')

import numpy as np

from blaze.bcolz import into, chunks
from blaze.expr import TableSymbol
from blaze.compute.core import compute


b = bcolz.ctable([[1, 2, 3], [1., 2., 3.]],
                 names=['a', 'b'])

t = TableSymbol('t', '{a: int32, b: float64}')

x = bcolz.ctable([[1,2,3,4,5], ['Alice','Bob','Charlie','Denis','Edith'], [100,-200,300,400,-500]], names=['id', 'name', 'amount'])
tx=TableSymbol('tx', '{id :int32, name:string, amount:int64}')


def test_chunks():
    assert len(list(chunks(b, chunksize=2))) == 2
    assert (next(chunks(b, chunksize=2)) == into(np.array(0), b)[:2]).all()


def test_reductions():
    assert compute(t.a.sum(), b) == 6
    assert compute(t.a.min(), b) == 1
    assert compute(t.a.max(), b) == 3
    assert compute(t.a.mean(), b) == 2.0
    assert abs(compute(t.a.std(), b) - np.std([1, 2, 3])) < 1e-5
    assert abs(compute(t.a.var(), b) - np.var([1, 2, 3])) < 1e-5
    assert abs(compute(t.a.std(unbiased=True), b) - np.std([1, 2, 3],
                                                           ddof=1)) < 1e-5
    assert abs(compute(t.a.var(unbiased=True), b) - np.var([1, 2, 3],
                                                           ddof=1)) < 1e-5
    assert compute(t.a.nunique(), b) == 3
    assert compute(t.nunique(), b) == 3
    assert len(list(compute(t.distinct(), b))) == 3
    assert len(list(compute(t.a.distinct(), b))) == 3


def test_selection_head():
    b = into(bcolz.ctable,
             ((i, i + 1, float(i)**2) for i in range(10000)),
             names=['a', 'b', 'c'])
    t = TableSymbol('t', '{a: int32, b: int32, c: float64}')

    assert compute((t.a < t.b).all(), b) == True
    assert list(compute(t[t.a < t.b].a.head(10), b)) == list(range(10))
    assert list(compute(t[t.a > t.b].a.head(10), b)) == []

    assert into([], compute(t[t.a + t.b > t.c], b)) == [(0, 1, 0),
                                                        (1, 2, 1),
                                                        (2, 3, 4)]
    assert len(compute(t[t.a + t.b > t.c].head(10), b)) # non-empty
    assert len(compute(t[t.a + t.b < t.c].head(10), b)) # non-empty


def test_selection_isnan():
    assert compute(t[t.a.isnan()].count(), b) == 0
    assert compute(t[~(t.a.isnan())].count(), b) == 3


def test_sample():
    test_data = x
    test_expr = tx

    result = compute(test_expr.sample(2), test_data)

    assert(len(result) == 2)

    for item in result:
        assert(item in test_data)

    
    result = compute(test_expr.sample(len(test_data)+1), test_data)
    assert(len(result) == len(test_data))
    assert(len(result) < (len(test_data)+1))

    for item in result:
        assert(item in test_data)
    
    #This test should give us repeated data
    result = compute(test_expr.sample(2*(len(test_data)), replacement=True), test_data)
    assert(len(result) == 2*(len(test_data)))

    for item in result:
        assert(item in test_data)
    
    #Test sampling from a single column from the array
    result = compute(test_expr['name'].sample(2), test_data)
    assert(len(result) == 2)
    assert(type(result) == type(test_data['name']))
    
    for item in result:
        assert(item in test_data['name'])
