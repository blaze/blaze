from __future__ import absolute_import, division, print_function

from contextlib import contextmanager
from operator import attrgetter
import pytest
import numpy as np
from pandas import DataFrame
from odo import odo, drop
from datashape.predicates import isscalar, iscollection, isrecord
from blaze.expr import symbol, by
from blaze.interactive import Data
from blaze.compute import compute
from blaze.expr.functions import sin, exp
import pandas as pd
import pandas.util.testing as tm


t = symbol('t', 'var * {amount: int64, id: int64, name: string}')

L = [[100, 1, 'Alice'],
     [200, 2, 'Bob'],
     [300, 3, 'Charlie'],
     [-400, 4, 'Dan'],
     [500, 5, 'Edith']]

df = DataFrame(L, columns=['amount', 'id', 'name'])


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


@pytest.mark.parametrize(
    ['string', 'func', 'source_name', 'source'],
    [
        (str(func(t)), func, source.__name__, source)
        for func, disabled in [
            (lambda t: t, set()),
            (lambda t: t.id, set()),
            (lambda t: abs(t['amount']), set()),
            (lambda t: t.id.max(), set()),
            (lambda t: t.amount.sum(), set()),
            (lambda t: t.amount.sum(keepdims=True), set()),
            (lambda t: t.amount.count(keepdims=True), set()),
            (lambda t: t.amount.nunique(keepdims=True), {mongo}),
            (lambda t: t.amount.nunique(keepdims=True), set()),
            (lambda t: t.amount.head(), {mongo}),
            (lambda t: t.amount + 1, set()),
            (lambda t: sin(t.amount), {sql, mongo}),
            (lambda t: exp(t.amount), {sql, mongo}),
            (lambda t: t.amount > 50, {mongo}),
            (lambda t: t[t.amount > 50], set()),
            (lambda t: t.like(name='Alic*'), set()),
            (lambda t: t.sort('name'), {bc}),
            (lambda t: t.sort('name', ascending=False), {bc}),
            (lambda t: t.head(3), set()),
            (lambda t: t.name.distinct(), set()),
            (lambda t: t[t.amount > 50]['name'], set()),
            (
                lambda t: t.id.map(lambda x: x + 1, schema='int64', name='id'),
                {sql, mongo}
            ),
            (lambda t: by(t.name, total=t.amount.sum()), set()),
            (lambda t: by(t.id, count=t.id.count()), set()),
            (lambda t: by(t[['id', 'amount']], count=t.id.count()), set()),
            (
                lambda t: by(t[['id', 'amount']], total=(t.amount + 1).sum()),
                {mongo}
            ),
            (
                lambda t: by(t[['id', 'amount']], n=t.name.nunique()),
                {mongo, bc}
            ),
            (lambda t: by(t.id, count=t.amount.count()), set()),
            (lambda t: by(t.id, n=t.id.nunique()), {mongo, bc}),
            # (lambda t: by(t, count=t.count()), []),
            # (lambda t: by(t.id, count=t.count()), []),

            # https://github.com/numpy/numpy/issues/3256
            (lambda t: t[['amount', 'id']], {numpy}),
            (lambda t: t[['id', 'amount']], {numpy, bc}),
            (lambda t: t[0], {sql, mongo, bc}),
            (lambda t: t[::2], {sql, mongo, bc}),
            (lambda t: t.id.utcfromtimestamp, {sql}),
            (lambda t: t.distinct().nrows, set()),
            (lambda t: t.nelements(axis=0), set()),
            (lambda t: t.nelements(axis=None), set()),
            (lambda t: t.amount.truncate(200), {sql})
        ]
        for source in sorted(
            possible_sources - set(disabled),
            key=attrgetter('__name__')
        )
    ]
)
def test_comprehensive(string, func, source_name, source):
    with source() as data:
        expr = func(t)
        model = compute(expr._subs({t: Data(df, t.dshape)}))
        T = Data(data)
        new_expr = expr._subs({t: T})
        if iscollection(expr.dshape):
            if isscalar(expr.dshape.measure):
                result = odo(new_expr, pd.Series)
                assert odo(result, set) == odo(model, set)
            else:
                result = odo(new_expr, pd.DataFrame)
                tm.assert_frame_equal(result.sort(), model.sort())
        elif isrecord(expr.dshape):
            assert odo(compute(new_expr), tuple) == odo(model, tuple)
        else:
            result = compute(new_expr)
            try:
                result = result.scalar()
            except AttributeError:
                pass
            assert result == model
