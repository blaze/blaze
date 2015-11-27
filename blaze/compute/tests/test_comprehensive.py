from __future__ import absolute_import, division, print_function

import pytest
import numpy as np
from pandas import DataFrame
from odo import resource, odo
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

x = odo(df, np.ndarray)

sources = {
    type(df): df,
    type(x): x
}

try:
    __import__('sqlalchemy')
    sql = resource('sqlite:///:memory:::accounts', dshape=t.dshape)
    odo(L, sql)
    sources.update({type(sql): sql})
except ImportError:
    sql = None


try:
    import bcolz
    bc = odo(df, bcolz.ctable)
    sources.update({type(bc): bc})
except ImportError:
    bc = None

try:
    import pymongo
except ImportError:
    pymongo = mongo = None


if pymongo is not None:
    try:
        db = pymongo.MongoClient().db

        try:
            coll = db._test_comprehensive
        except AttributeError:
            coll = db['_test_comprehensive']

        coll.drop()
        mongo = odo(df, coll)
        sources.update({type(mongo): mongo})
    except pymongo.errors.ConnectionFailure:
        mongo = None


@pytest.mark.parametrize(
    ['func', 'string', 'typ'],
    [
        (func, str(func(t)), k)
        for func, disabled in [
            (lambda t: t, []),
            (lambda t: t.id, []),
            (lambda t: abs(t['amount']), []),
            (lambda t: t.id.max(), []),
            (lambda t: t.amount.sum(), []),
            (lambda t: t.amount.sum(keepdims=True), []),
            (lambda t: t.amount.count(keepdims=True), []),
            (lambda t: t.amount.nunique(keepdims=True), [type(mongo)]),
            (lambda t: t.amount.nunique(keepdims=True), []),
            (lambda t: t.amount.head(), [type(mongo)]),
            (lambda t: t.amount + 1, []),
            (lambda t: sin(t.amount), [type(sql), type(mongo)]),
            (lambda t: exp(t.amount), [type(sql), type(mongo)]),
            (lambda t: t.amount > 50, [type(mongo)]),
            (lambda t: t[t.amount > 50], []),
            (lambda t: t.like(name='Alic*'), []),
            (lambda t: t.sort('name'), [type(bc)]),
            (lambda t: t.sort('name', ascending=False), [type(bc)]),
            (lambda t: t.head(3), []),
            (lambda t: t.name.distinct(), []),
            (lambda t: t[t.amount > 50]['name'], []),
            (
                lambda t: t.id.map(lambda x: x + 1, schema='int64', name='id'),
                [type(sql), type(mongo)]
            ),
            (lambda t: by(t.name, total=t.amount.sum()), []),
            (lambda t: by(t.id, count=t.id.count()), []),
            (lambda t: by(t[['id', 'amount']], count=t.id.count()), []),
            (
                lambda t: by(t[['id', 'amount']], total=(t.amount + 1).sum()),
                [type(mongo)]
            ),
            (
                lambda t: by(t[['id', 'amount']], n=t.name.nunique()),
                [type(mongo), type(bc)]
            ),
            (lambda t: by(t.id, count=t.amount.count()), []),
            (lambda t: by(t.id, n=t.id.nunique()), [type(mongo), type(bc)]),
            # (lambda t: by(t, count=t.count()), []),
            # (lambda t: by(t.id, count=t.count()), []),

            # https://github.com/numpy/numpy/issues/3256
            (lambda t: t[['amount', 'id']], [type(x)]),
            (lambda t: t[['id', 'amount']], [type(x), type(bc)]),
            (lambda t: t[0], list(map(type, [sql, mongo, bc]))),
            (lambda t: t[::2], list(map(type, [sql, mongo, bc]))),
            (lambda t: t.id.utcfromtimestamp, [type(sql)]),
            (lambda t: t.distinct().nrows, []),
            (lambda t: t.nelements(axis=0), []),
            (lambda t: t.nelements(axis=None), []),
            (lambda t: t.amount.truncate(200), [type(sql)])
        ]
        for k in sorted(set(sources) - set(disabled), key=str)
    ]
)
def test_new_base(func, string, typ):
    expr = func(t)
    model = compute(expr._subs({t: Data(df, t.dshape)}))
    source = sources[typ]
    T = Data(source)
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
