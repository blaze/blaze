from __future__ import absolute_import, division, print_function

import pytest
import operator

from datashape import discover, dshape

from blaze.compute.core import (compute_up, compute, bottom_up_until_type_break,
                                top_then_bottom_then_top_again_etc,
                                swap_resources_into_scope)
from blaze.expr import by, symbol, Expr, Symbol
from blaze.dispatch import dispatch
from blaze.compatibility import raises, reduce
from blaze.utils import example
from blaze.interactive import into

import pandas as pd
import numpy as np
import dask.array as da


def test_errors():
    t = symbol('t', 'var * {foo: int}')
    with raises(NotImplementedError):
        compute_up(by(t, count=t.count()), 1)


def test_optimize():
    class Foo(object):
        pass

    s = symbol('s', '5 * {x: int, y: int}')

    @dispatch(Expr, Foo)
    def compute_down(expr, foo):
        return str(expr)

    assert compute(s.x * 2, Foo()) == "s.x * 2"

    @dispatch(Expr, Foo)
    def optimize(expr, foo):
        return expr + 1

    assert compute(s.x * 2, Foo()) == "(s.x * 2) + 1"


def test_bottom_up_until_type_break():

    s = symbol('s', 'var * {name: string, amount: int}')
    data = np.array([('Alice', 100), ('Bob', 200), ('Charlie', 300)],
                    dtype=[('name', 'S7'), ('amount', 'i4')])

    e = (s.amount + 1).distinct()
    expr, scope = bottom_up_until_type_break(e, {s: data})
    amount = symbol('amount', 'var * int64', token=expr._token)
    assert expr.isidentical(amount)
    assert len(scope) == 1
    assert amount in scope
    assert (scope[amount] == np.array([101, 201, 301], dtype='i4')).all()

    # This computation has a type change midstream, so we stop and get the
    # unfinished computation.

    e = s.amount.sum() + 1
    expr, scope = bottom_up_until_type_break(e, {s: data})
    amount_sum = symbol('amount_sum', 'int64', token=expr.lhs._token)
    assert expr.isidentical(amount_sum + 1)
    assert len(scope) == 1
    assert amount_sum in scope
    assert scope[amount_sum] == 600

    # ensure that we work on binops with one child
    x = symbol('x', 'real')
    expr, scope = bottom_up_until_type_break(x + x, {x: 1})
    assert len(scope) == 1
    x2 = list(scope.keys())[0]
    assert isinstance(x2, Symbol)
    assert isinstance(expr, Symbol)
    assert scope[x2] == 2


def test_top_then_bottom_then_top_again_etc():
    s = symbol('s', 'var * {name: string, amount: int32}')
    data = np.array([('Alice', 100), ('Bob', 200), ('Charlie', 300)],
                    dtype=[('name', 'S7'), ('amount', 'i4')])

    e = s.amount.sum() + 1
    assert top_then_bottom_then_top_again_etc(e, {s: data}) == 601


def test_swap_resources_into_scope():

    from blaze import data
    t = data([1, 2, 3], dshape='3 * int', name='t')
    expr, scope = swap_resources_into_scope(t.head(2), {t: t.data})

    assert t._resources()
    assert not expr._resources()

    assert t not in scope


def test_compute_up_on_dict():
    d = {'a': [1, 2, 3], 'b': [4, 5, 6]}

    assert str(discover(d)) == str(dshape('{a: 3 * int64, b: 3 * int64}'))

    s = symbol('s', discover(d))

    assert compute(s.a, {s: d}) == [1, 2, 3]


def test_pre_compute_on_multiple_datasets_is_selective():
    from odo import CSV
    from blaze import data
    from blaze.cached import CachedDataset

    df = pd.DataFrame([[1, 'Alice',   100],
                         [2, 'Bob',    -200],
                         [3, 'Charlie', 300],
                         [4, 'Denis',   400],
                         [5, 'Edith',  -500]], columns=['id', 'name', 'amount'])
    iris = CSV(example('iris.csv'))
    dset = CachedDataset({'df': df, 'iris': iris})

    d = data(dset)
    assert str(compute(d.df.amount)) == str(df.amount)


def test_raises_on_valid_expression_but_no_implementation():
    class MyExpr(Expr):
        __slots__ = '_hash', '_child'

        @property
        def dshape(self):
            return self._child.dshape

    t = symbol('t', 'var * {amount: real}')
    expr = MyExpr(t.amount)
    df = [(1.0,), (2.0,), (3.0,)]
    with pytest.raises(NotImplementedError):
        compute(expr, df)


@pytest.mark.parametrize('n', range(2, 11))
def test_simple_add(n):
    x = symbol('x', 'int')
    expr = reduce(operator.add, [x] * n)
    assert compute(expr, 1) == n


@pytest.mark.parametrize('data,expr,ret_type,exp_type', [
    (1, symbol('x', 'int'), 'native', int),
    (1, symbol('x', 'int'), 'core', int),
    # use dask array to test core since isn't core type
    (into(da.core.Array, [1, 2], chunks=(10,)), symbol('x', '2 * int'), 'core', pd.Series),  # test 1-d to series
    (into(da.core.Array, [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}], chunks=(10,10)), symbol('x', '2 * {a: int, b: int}'), 'core', pd.DataFrame),  # test 2-d tabular to dataframe
    (into(da.core.Array, [[1, 2], [3, 4]], chunks=(10, 10)), symbol('x', '2 *  2 * int'), 'core', np.ndarray),  # test 2-d non tabular to ndarray
    ([1, 2], symbol('x', '2 * int') , tuple, tuple)
])
def test_compute_return_type(data, expr, ret_type, exp_type):
    assert isinstance(compute(expr, data, return_type=ret_type), exp_type)
