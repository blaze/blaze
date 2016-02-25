from __future__ import absolute_import, division, print_function

import numpy as np
from pandas import DataFrame
import numpy as np
from odo import into
from datashape.predicates import isscalar, iscollection, isrecord
from blaze.expr import symbol, by
from blaze.interactive import data
from blaze.compute import compute
from blaze.expr.functions import sin, exp


sources = []

t = symbol('t', 'var * {amount: int64, id: int64, name: string}')

L = [[ 100, 1, 'Alice'],
     [ 200, 2, 'Bob'],
     [ 300, 3, 'Charlie'],
     [-400, 4, 'Dan'],
     [ 500, 5, 'Edith']]

df = DataFrame(L, columns=['amount', 'id', 'name'])

x = into(np.ndarray, df)

sources = [df, x]

try:
    import sqlalchemy
    sql = data('sqlite:///:memory:::accounts', dshape=t.dshape)
    into(sql, L)
    sources.append(sql)
except:
    sql = None


try:
    import bcolz
    bc = into(bcolz.ctable, df)
    sources.append(bc)
except ImportError:
    bc = None

try:
    import pymongo
except ImportError:
    pymongo = mongo = None
if pymongo:

    try:
        db = pymongo.MongoClient().db

        try:
            coll = db._test_comprehensive
        except AttributeError:
            coll = db['_test_comprehensive']

        coll.drop()
        mongo = into(coll, df)
        sources.append(mongo)
    except pymongo.errors.ConnectionFailure:
        mongo = None

# {expr: [list-of-exclusions]}
expressions = {
        t: [],
        t['id']: [],
        abs(t['amount']): [],
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
        t[t.name.like('Alic*')]: [],
        t.sort('name'): [bc],
        t.sort('name', ascending=False): [bc],
        t.head(3): [],
        t.name.distinct(): [],
        t[t.amount > 50]['name']: [], # odd ordering issue
        t.id.map(lambda x: x + 1, schema='int64', name='id'): [sql, mongo],
        t[t.amount > 50]['name']: [],
        by(t.name, total=t.amount.sum()): [],
        by(t.id, count=t.id.count()): [],
        by(t[['id', 'amount']], count=t.id.count()): [],
        by(t[['id', 'amount']], total=(t.amount + 1).sum()): [mongo],
        by(t[['id', 'amount']], n=t.name.nunique()): [mongo, bc],
        by(t.id, count=t.amount.count()): [],
        by(t.id, n=t.id.nunique()): [mongo, bc],
        # by(t, count=t.count()): [],
        # by(t.id, count=t.count()): [],
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
            # and list(a.dtypes) == list(b.dtypes)
            and into(set, into(list, a)) == into(set, into(list, b)))


def typename(obj):
    return type(obj).__name__


def test_base():
    for expr, exclusions in expressions.items():
        if iscollection(expr.dshape):
            model = into(DataFrame, into(np.ndarray, expr._subs({t: data(base, t.dshape)})))
        else:
            model = compute(expr._subs({t: data(base, t.dshape)}))
        print('\nexpr: %s\n' % expr)
        for source in sources:
            if id(source) in map(id, exclusions):
                continue
            print('%s <- %s' % (typename(model), typename(source)))
            T = data(source)
            if iscollection(expr.dshape):
                result = into(type(model), expr._subs({t: T}))
                if isscalar(expr.dshape.measure):
                    assert set(into(list, result)) == set(into(list, model))
                else:
                    assert df_eq(result, model)
            elif isrecord(expr.dshape):
                result = compute(expr._subs({t: T}))
                assert into(tuple, result) == into(tuple, model)
            else:
                result = compute(expr._subs({t: T}))
                try:
                    result = result.scalar()
                except AttributeError:
                    pass
                assert result == model
