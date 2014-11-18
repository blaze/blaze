from __future__ import absolute_import, division, print_function

import numpy as np
from pandas import DataFrame
import numpy as np
import bcolz
from datashape.predicates import isscalar, iscollection, isrecord
from blaze.expr import Symbol, by
from blaze.api import Data, into
from blaze.compute import compute
from blaze.expr.functions import sin, exp
from blaze.sql import SQL


sources = []

t = Symbol('t', 'var * {amount: int64, id: int64, name: string}')

L = [[100, 1, 'Alice'],
     [200, 2, 'Bob'],
     [300, 3, 'Charlie'],
     [400, 4, 'Dan'],
     [500, 5, 'Edith']]

df = DataFrame(L, columns=['amount', 'id', 'name'])

x = into(np.ndarray, df)

bc = into(bcolz.ctable, df)

sql = SQL('sqlite:///:memory:', 'accounts', schema=t.schema)
sql.extend(L)

sources = [df, x, bc, sql]

try:
    import pymongo
except ImportError:
    pymongo = mongo = None
if pymongo:
    from blaze.mongo import *
    try:
        db = pymongo.MongoClient().db
        db._test_comprehensive.drop()
        mongo = into(db._test_comprehensive, df)
        sources.append(mongo)
    except pymongo.errors.ConnectionFailure:
        mongo = None

# {expr: [list-of-exclusions]}
expressions = {
        t: [],
        t['id']: [],
        t.id.max(): [],
        t.amount.sum(): [],
        t.amount.sum(keepdims=True): [],
        t.amount.count(keepdims=True): [],
        t.amount.nunique(keepdims=True): [mongo],
        t.amount.nunique(): [],
        t.amount.head(): [],
        t.amount + 1: [mongo],
        sin(t.amount): [sql, mongo], # sqlite doesn't support trig
        exp(t.amount): [sql, mongo],
        t.amount > 50: [mongo],
        t[t.amount > 50]: [],
        t.like(name='Alic*'): [],
        t.sort('name'): [bc],
        t.sort('name', ascending=False): [bc],
        t.head(3): [],
        t.name.distinct(): [],
        t[t.amount > 50]['name']: [], # odd ordering issue
        t.id.map(lambda x: x + 1, schema='int', name='id'): [sql, mongo],
        t[t.amount > 50]['name']: [],
        by(t.name, total=t.amount.sum()): [],
        by(t.id, count=t.id.count()): [],
        by(t[['id', 'amount']], count=t.id.count()): [],
        by(t[['id', 'amount']], total=(t.amount + 1).sum()): [mongo],
        by(t[['id', 'amount']], n=t.name.nunique()): [mongo, bc],
        by(t.id, count=t.amount.count()): [],
        by(t.id, n=t.id.nunique()): [mongo],
        # by(t, t.count()): [],
        # by(t.id, t.count()): [df],
        t[['amount', 'id']]: [x], # https://github.com/numpy/numpy/issues/3256
        t[['id', 'amount']]: [x, bc], # bcolz sorting
        t[0]: [sql, mongo, bc],
        t[::2]: [sql, mongo, bc],
        t.id.utcfromtimestamp: [sql],
        t.distinct().nrows: [],
        t.nelements(axis=0): [],
        t.nelements(axis=None): [],
        t.amount.truncate(200): [sql]
        }

base = df


def df_eq(a, b):
    return (list(a.columns) == list(b.columns)
            and list(a.dtypes) == list(b.dtypes)
            and into(set, into(list, a)) == into(set, into(list, b)))


def typename(obj):
    return type(obj).__name__


def test_base():
    for expr, exclusions in expressions.items():
        model = compute(expr._subs({t: Data(base, t.dshape)}))
        print('\nexpr: %s\n' % expr)
        for source in sources:
            if id(source) in map(id, exclusions):
                continue
            print('%s <- %s' % (typename(model), typename(source)))
            T = Data(source)
            if iscollection(expr.dshape):
                result = into(model, expr._subs({t: T}))
                if isscalar(expr.dshape.measure):
                    assert set(into([], result)) == set(into([], model))
                else:
                    assert df_eq(result, model)
            elif isrecord(expr.dshape):
                result = compute(expr._subs({t: T}))
                assert into(tuple, result) == into(tuple, model)
            else:
                result = compute(expr._subs({t: T}))
                assert result == model
