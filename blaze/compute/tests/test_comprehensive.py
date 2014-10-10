from __future__ import absolute_import, division, print_function

import numpy as np
from pandas import DataFrame
import numpy as np
import bcolz
from datashape.predicates import isscalar, iscollection, isrecord
from blaze.expr import TableSymbol, by
from blaze.api.into import into
from blaze.api.table import Table
from blaze.compute import compute
from blaze.expr.functions import sin, exp
from blaze.sql import SQL


sources = []

t = TableSymbol('t', '{amount: int64, id: int64, name: string}')

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
        by(t.name, t.amount.sum()): [],
        by(t.id, t.id.count()): [],
        by(t[['id', 'amount']], t.id.count()): [],
        by(t[['id', 'amount']], (t.amount + 1).sum()): [mongo],
        by(t[['id', 'amount']], t.name.nunique()): [mongo],
        by(t.id, t.amount.count()): [],
        by(t.id, t.id.nunique()): [mongo],
        # by(t, t.count()): [],
        # by(t.id, t.count()): [df],
        t[['amount', 'id']]: [x], # https://github.com/numpy/numpy/issues/3256
        t[['id', 'amount']]: [x, bc], # bcolz sorting
        t[0]: [sql, mongo],
        t[::2]: [sql, mongo],
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
        model = compute(expr._subs({t: Table(base, t.schema)}))
        print('\nexpr: %s\n' % expr)
        for source in sources:
            if id(source) in map(id, exclusions):
                continue
            print('%s <- %s' % (typename(model), typename(source)))
            T = Table(source)
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
