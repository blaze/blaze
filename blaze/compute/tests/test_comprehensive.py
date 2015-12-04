from __future__ import absolute_import, division, print_function

from contextlib import contextmanager
from operator import attrgetter
import pytest
import numpy as np
from odo import odo, drop
from datashape.predicates import isscalar, iscollection, isrecord
from blaze.expr import symbol, by
from blaze.interactive import Data
from blaze.compute import compute
from blaze.expr.functions import sin, exp
import pandas as pd
import pandas.util.testing as tm


t = symbol('t', 'var * {amount: int64, id: int64, name: string}')


L = [
    [100, 1, 'Alice'],
    [200, 2, 'Bob'],
    [300, 3, 'Charlie'],
    [-400, 4, 'Dan'],
    [500, 5, 'Edith']
]


df = pd.DataFrame(L, columns=['amount', 'id', 'name'])


@contextmanager
def numpy():
    yield odo(df, np.ndarray)


@contextmanager
def sql():
    try:
        import sqlalchemy
    except ImportError as e:
        pytest.skip(str(e))
    else:
        table = odo(L, 'sqlite:///:memory:::accounts', dshape=t.dshape)
        try:
            yield table
        finally:
            drop(table)


@contextmanager
def bc():
    try:
        import bcolz
    except ImportError as e:
        pytest.skip(str(e))
    else:
        yield odo(df, bcolz.ctable)


@contextmanager
def mongo():
    try:
        import pymongo
    except ImportError as e:
        pytest.skip(str(e))
    else:
        try:
            db = pymongo.MongoClient().db

            try:
                coll = db._test_comprehensive
            except AttributeError:
                coll = db['_test_comprehensive']

            coll.drop()
            yield odo(df, coll)
        except pymongo.errors.ConnectionFailure as e:
            pytest.skip(str(e))


possible_sources = frozenset({numpy, sql, bc, mongo})


exprs = [
    (t, set()),
    (t.id, set()),
    (abs(t['amount']), set()),
    (t.id.max(), set()),
    (t.amount.sum(), set()),
    (t.amount.sum(keepdims=True), set()),
    (t.amount.count(keepdims=True), set()),
    (t.amount.nunique(keepdims=True), {mongo}),
    (t.amount.nunique(keepdims=True), set()),
    (t.amount.head(), {mongo}),
    (t.amount + 1, set()),
    (sin(t.amount), {sql, mongo}),
    (exp(t.amount), {sql, mongo}),
    (t.amount > 50, {mongo}),
    (t[t.amount > 50], set()),
    (t.like(name='Alic*'), set()),
    (t.sort('name'), {bc}),
    (t.sort('name', ascending=False), {bc}),
    (t.head(3), set()),
    (t.name.distinct(), set()),
    (t[t.amount > 50]['name'], set()),
    (
        t.id.map(lambda x: x + 1, schema='int64', name='id'),
        {sql, mongo}
    ),
    (by(t.name, total=t.amount.sum()), set()),
    (by(t.id, count=t.id.count()), set()),
    (by(t[['id', 'amount']], count=t.id.count()), set()),
    (by(t[['id', 'amount']], total=(t.amount + 1).sum()), {mongo}),
    (by(t[['id', 'amount']], n=t.name.nunique()), {mongo, bc}),
    (by(t.id, count=t.amount.count()), set()),
    (by(t.id, n=t.id.nunique()), {mongo, bc}),
    # (lambda t: by(t, count=t.count()), []),
    # (lambda t: by(t.id, count=t.count()), []),

    # https://github.com/numpy/numpy/issues/3256
    (t[['amount', 'id']], {numpy}),
    (t[['id', 'amount']], {numpy, bc}),
    (t[0], {sql, mongo, bc}),
    (t[::2], {sql, mongo, bc}),
    (t.id.utcfromtimestamp, {sql}),
    (t.distinct().nrows, set()),
    (t.nelements(axis=0), set()),
    (t.nelements(axis=None), set()),
    (t.amount.truncate(200), {sql})
]


@pytest.mark.parametrize(
    ['string', 'source_name', 'expr', 'source'],
    [
        (str(expr), source.__name__, expr, source)
        for expr, disabled in exprs
        for source in sorted(
            possible_sources - set(disabled),
            key=attrgetter('__name__')
        )
    ]
)
def test_comprehensive(string, source_name, expr, source):
    with source() as data:
        model = compute(expr._subs({t: Data(df, t.dshape)}))
        T = Data(data)
        new_expr = expr._subs({t: T})
        if iscollection(expr.dshape):
            if isscalar(expr.dshape.measure):
                result = odo(new_expr, pd.Series)
                assert odo(result, set) == odo(model, set)
            else:
                result = odo(new_expr, pd.DataFrame)
                lhs = result.sort(expr.fields[0]).reset_index(drop=True)
                rhs = model.sort(expr.fields[0]).reset_index(drop=True)

                # NOTE: sometimes we get int64 vs int32 and we don't want that
                # to fail, or do we?
                tm.assert_frame_equal(lhs, rhs, check_dtype=False)
        elif isrecord(expr.dshape):
            assert odo(compute(new_expr), tuple) == odo(model, tuple)
        else:
            result = compute(new_expr)
            try:
                result = result.scalar()
            except AttributeError:
                pass
            assert result == model
