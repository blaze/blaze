from __future__ import absolute_import, division, print_function

import pytest
import pandas as pd
from operator import (add, sub, mul, floordiv, mod, pow, truediv, eq, ne, lt,
                      gt, le, ge, getitem)

from functools import partial
from datetime import datetime
import datashape
from datashape.predicates import iscollection, isscalar
from blaze import CSV, Table
from blaze.expr import (TableSymbol, projection, Field, selection, Broadcast,
                        join, cos, by, exp, distinct, Apply,
                        broadcast, eval_str, merge, common_subexpression, sum,
                        Label, ReLabel, Head, Sort, any, summary,
                        Summary, count, symbol, Field, discover,
                        max, min, label, Symbol, transform
                        )
from blaze.compatibility import PY3, builtins
from blaze.utils import raises, tmpfile
from datashape import dshape, var, int32, int64, Record, DataShape
from toolz import identity, first
import numpy as np


def test_dshape():
    t = TableSymbol('t', '{name: string, amount: int}')
    assert t.dshape == dshape('var * {name: string, amount: int}')


def test_length():
    t = TableSymbol('t', '10 * {name: string, amount: int}')
    s = TableSymbol('s', '{name:string, amount:int}')
    assert t.dshape == dshape('10 * {name: string, amount: int}')
    assert len(t) == 10
    assert len(t.name) == 10
    assert len(t[['name']]) == 10
    assert len(t.sort('name')) == 10
    assert len(t.head(5)) == 5
    assert len(t.head(50)) == 10
    with pytest.raises(ValueError):
        len(s)


def test_tablesymbol_eq():
    assert not (TableSymbol('t', '{name: string}')
             == TableSymbol('v', '{name: string}'))


def test_table_name():
    t = TableSymbol('t', '10 * {people: string, amount: int}')
    r = TableSymbol('r', 'int64')
    with pytest.raises(AttributeError):
        t.name
    with pytest.raises(AttributeError):
        r.name


def test_shape():
    t = TableSymbol('t', '{name: string, amount: int}')
    assert t.shape
    assert isinstance(t.shape, tuple)
    assert len(t.shape) == 1


def test_eq():
    assert TableSymbol('t', '{a: string, b: int}').isidentical(
            TableSymbol('t', '{a: string, b: int}'))
    assert not TableSymbol('t', '{b: string, a: int}').isidentical(
            TableSymbol('t', '{a: string, b: int}'))


def test_arithmetic():
    t = TableSymbol('t', '{x: int, y: int, z: int}')
    x, y, z = t['x'], t['y'], t['z']
    exprs = [x + 1, x + y, 1 + y,
             x - y, 1 - x, x - 1,
             x ** y, x ** 2, 2 ** x,
             x * y, x ** 2, 2 ** x,
             x / y, x / 2, 2 / x,
             x % y, x % 2, 2 % x]


def test_column():
    t = TableSymbol('t', '{name: string, amount: int}')
    assert t.fields== ['name', 'amount']

    assert eval(str(t.name)) == t.name
    assert str(t.name) == "t.name"
    with pytest.raises(AttributeError):
        t.name.balance
    with pytest.raises((NotImplementedError, ValueError)):
        getitem(t, set('balance'))


def test_symbol_projection_failures():
    t = TableSymbol('t', '10 * {name: string, amount: int}')
    with pytest.raises(ValueError):
        t._project(['name', 'id'])
    with pytest.raises(AttributeError):
        t.foo
    with pytest.raises(TypeError):
        t._project(t.dshape)


def test_Projection():
    t = TableSymbol('t', '{name: string, amount: int, id: int32}')
    p = projection(t, ['amount', 'name'])
    assert p.schema == dshape('{amount: int32, name: string}')
    print(t['amount'].dshape)
    print(dshape('var * int32'))
    assert t['amount'].dshape == dshape('var * int32')
    assert t['amount']._name == 'amount'

    assert eval(str(p)).isidentical(p)
    assert p._project(['amount','name']) == p[['amount','name']]
    with pytest.raises(ValueError):
        p._project('balance')


def test_Projection_retains_shape():
    t = TableSymbol('t', '5 * {name: string, amount: int, id: int32}')

    assert t[['name', 'amount']].dshape == \
            dshape('5 * {name: string, amount: int}')


def test_indexing():
    t = TableSymbol('t', '{name: string, amount: int, id: int}')
    assert t[['amount', 'id']] == projection(t, ['amount', 'id'])
    assert t['amount'].isidentical(Field(t, 'amount'))


def test_relational():
    t = TableSymbol('t', '{name: string, amount: int, id: int}')

    r = (t['name'] == 'Alice')

    assert 'bool' in str(r.dshape)
    assert r._name


def test_selection():
    t = TableSymbol('t', '{name: string, amount: int, id: int}')

    s = selection(t, t['name'] == 'Alice')
    f = selection(t, t['id'] > t['amount'])
    p = t[t['amount'] > 100]
    with pytest.raises(ValueError):
        selection(t, p)

    assert s.dshape == t.dshape


def test_selection_typecheck():
    t = TableSymbol('t', '{name: string, amount: int, id: int}')

    assert raises(TypeError, lambda: t[t['amount'] + t['id']])
    assert raises(TypeError, lambda: t[t['name']])


def test_selection_by_indexing():
    t = TableSymbol('t', '{name: string, amount: int, id: int}')

    result = t[t['name'] == 'Alice']

    assert t.schema == result.schema
    assert 'Alice' in str(result)


def test_selection_by_getattr():
    t = TableSymbol('t', '{name: string, amount: int, id: int}')

    result = t[t.name == 'Alice']

    assert t.schema == result.schema
    assert 'Alice' in str(result)


def test_selection_path_check():
    t = TableSymbol('t', '{name: string, amount: int, id: int}')
    t2 = t[t.name == 'Alice']
    t3 = t2[t2.amount > 0]


def test_path_issue():
    t = TableSymbol('t', "{topic: string, word: string, result: ?float64}")
    t2 = transform(t, sizes=t.result.map(lambda x: (x - MIN)*10/(MAX - MIN),
                                         schema='float64', name='size'))

    assert builtins.any(t2.sizes.isidentical(node) for node in t2.children)


def test_getattr_doesnt_override_properties():
    t = TableSymbol('t', '{_subs: string, schema: string}')
    assert callable(t._subs)
    assert isinstance(t.schema, DataShape)


def test_dir_contains_columns():
    t = TableSymbol('t', '{name: string, amount: int, id: int}')
    result = dir(t)
    columns_set = set(t.fields)
    assert set(result) & columns_set == columns_set


def test_selection_consistent_children():
    t = TableSymbol('t', '{name: string, amount: int, id: int}')

    expr = t['name'][t['amount'] < 0]

    assert list(expr.fields) == ['name']


def test_str():
    import re
    t = TableSymbol('t', '{name: string, amount: int, id: int}')
    expr = t[t['amount'] < 0]['id'] * 2
    assert '<class' not in str(expr)
    assert not re.search('0x[0-9a-f]+', str(expr))

    assert eval(str(expr)) == expr

    assert '*' in repr(expr)


def test_join():
    t = TableSymbol('t', '{name: string, amount: int}')
    s = TableSymbol('t', '{name: string, id: int}')
    r = TableSymbol('r', '{name: string, amount: int}')
    q = TableSymbol('q', '{name: int}')

    j = join(t, s, 'name', 'name')

    assert j.schema == dshape('{name: string, amount: int, id: int}')

    assert join(t, s, 'name') == join(t, s, 'name')

    assert join(t, s, 'name').on_left == 'name'
    assert join(t, s, 'name').on_right == 'name'

    assert join(t, r, ('name', 'amount')).on_left == ['name', 'amount']
    with pytest.raises(TypeError):
        join(t, q, 'name')
    with pytest.raises(ValueError):
        join(t, s, how='upside_down')


def test_join_different_on_right_left_columns():
    t = TableSymbol('t', '{x: int, y: int}')
    s = TableSymbol('t', '{a: int, b: int}')
    j = join(t, s, 'x', 'a')
    assert j.on_left == 'x'
    assert j.on_right == 'a'


def test_joined_column_first_in_schema():
    t = TableSymbol('t', '{x: int, y: int, z: int}')
    s = TableSymbol('s', '{w: int, y: int}')

    assert join(t, s).schema == dshape('{y: int, x: int, z: int, w: int}')


def test_outer_join():
    t = TableSymbol('t', '{name: string, amount: int}')
    s = TableSymbol('t', '{name: string, id: int}')

    jleft = join(t, s, 'name', 'name', how='left')
    jright = join(t, s, 'name', 'name', how='right')
    jinner = join(t, s, 'name', 'name', how='inner')
    jouter = join(t, s, 'name', 'name', how='outer')

    js = [jleft, jright, jinner, jouter]

    assert len(set(js)) == 4  # not equal

    assert jinner.schema == dshape('{name: string, amount: int, id: int}')
    assert jleft.schema == dshape('{name: string, amount: int, id: ?int}')
    assert jright.schema == dshape('{name: string, amount: ?int, id: int}')
    assert jouter.schema == dshape('{name: string, amount: ?int, id: ?int}')

    # Default behavior
    assert join(t, s, 'name', 'name', how='inner') == \
            join(t, s, 'name', 'name')


def test_join_default_shared_columns():
    t = TableSymbol('t', '{name: string, amount: int}')
    s = TableSymbol('t', '{name: string, id: int}')
    assert join(t, s) == join(t, s, 'name', 'name')


def test_multi_column_join():
    a = TableSymbol('a', '{x: int, y: int, z: int}')
    b = TableSymbol('b', '{w: int, x: int, y: int}')
    j = join(a, b, ['x', 'y'])

    assert set(j.fields) == set('wxyz')

    assert j.on_left == j.on_right == ['x', 'y']
    assert hash(j)

    assert j.fields == ['x', 'y', 'z', 'w']


def test_traverse():
    t = TableSymbol('t', '{name: string, amount: int}')
    assert t in list(t._traverse())

    expr = t.amount.sum()
    trav = list(expr._traverse())
    assert builtins.any(t.amount.isidentical(x) for x in trav)


def test_unary_ops():
    t = TableSymbol('t', '{name: string, amount: int}')
    expr = cos(exp(t['amount']))
    assert 'cos' in str(expr)

    assert '~' in str(~(t.amount > 0))


def test_reduction():
    t = TableSymbol('t', '{name: string, amount: int32}')
    r = sum(t['amount'])
    assert r.dshape in (dshape('int64'),
                        dshape('{amount: int64}'),
                        dshape('{amount_sum: int64}'))

    assert 'amount' not in str(t.count().dshape)

    assert t.count().dshape[0] in (int32, int64)

    assert 'int' in str(t.count().dshape)
    assert 'int' in str(t.nunique().dshape)
    assert 'string' in str(t['name'].max().dshape)
    assert 'string' in str(t['name'].min().dshape)
    assert 'string' not in str(t.count().dshape)

    t = TableSymbol('t', '{name: string, amount: real, id: int}')

    assert 'int' in str(t['id'].sum().dshape)
    assert 'int' not in str(t['amount'].sum().dshape)


def test_reduction_name():
    t = TableSymbol('t', '{name: string, amount: int32, id: int32}')
    assert (t.amount + t.id).sum()._name


def test_max_min_class():
    t = TableSymbol('t', '{name: string, amount: int32}')
    assert str(max(t).dshape) == '{name: string, amount: int32}'
    assert str(min(t).dshape) == '{name: string, amount: int32}'


@pytest.fixture
def symsum():
    t = TableSymbol('t', '{name: string, amount: int32}')
    return t, t.amount.sum()


@pytest.fixture
def ds():
    return dshape("var * { "
            "transaction_key : int64, "
            "user_from_key : int64, "
            "user_to_key : int64, "
            "date : int64, "
            "value : float64 "
            "}")


def test_discover_dshape_symbol(ds):
    t_ds = TableSymbol('t', dshape=ds)
    assert t_ds.fields is not None

    t_sch = TableSymbol('t', dshape=ds.subshape[0])
    assert t_sch.fields is not None

    assert t_ds.isidentical(t_sch)


class TestScalarArithmetic(object):
    ops = {'+': add, '-': sub, '*': mul, '/': truediv, '//': floordiv, '%': mod,
           '**': pow, '==': eq, '!=': ne, '<': lt, '>': gt, '<=': le, '>=': ge}

    def test_scalar_arith(self, symsum):
        def runner(f):
            result = f(r, 1)
            assert eval('r %s 1' % op).isidentical(result)

            a = f(r, r)
            b = eval('r %s r' % op)
            assert a is b or a.isidentical(b)

            result = f(1, r)
            assert eval('1 %s r' % op).isidentical(result)

        t, r = symsum
        r = t.amount.sum()
        for op, f in self.ops.items():
            runner(f)

    def test_scalar_usub(self, symsum):
        t, r = symsum
        result = -r
        assert eval(str(result)).isidentical(result)

    @pytest.mark.xfail
    def test_scalar_uadd(self, symsum):
        t, r = symsum
        +r


def test_summary():
    t = TableSymbol('t', '{id: int32, name: string, amount: int32}')
    s = summary(total=t.amount.sum(), num=t.id.count())
    assert s.dshape == dshape('{num: int32, total: int64}')
    assert hash(s)
    assert eval(str(s)).isidentical(s)

    assert 'summary(' in str(s)
    assert 'total=' in str(s)
    assert 'num=' in str(s)
    assert str(t.amount.sum()) in str(s)

    assert not summary(total=t.amount.sum())._child.isidentical(
            t.amount.sum())
    assert iscollection(summary(total=t.amount.sum() + 1)._child.dshape)


def test_reduction_arithmetic():
    t = TableSymbol('t', '{id: int32, name: string, amount: int32}')
    expr = t.amount.sum() + 1
    assert eval(str(expr)).isidentical(expr)


def test_Distinct():
    t = TableSymbol('t', '{name: string, amount: int32}')
    r = distinct(t['name'])
    print(r.dshape)
    assert r.dshape  == dshape('var * string')
    assert r._name == 'name'

    r = t.distinct()
    assert r.dshape  == t.dshape


def test_by():
    t = TableSymbol('t', '{name: string, amount: int32, id: int32}')
    r = by(t['name'], total=sum(t['amount']))

    print(r.schema)
    assert isinstance(r.schema[0], Record)
    assert str(r.schema[0]['name']) == 'string'


def test_by_summary():
    t = TableSymbol('t', '{name: string, amount: int32, id: int32}')
    a = by(t['name'], sum=sum(t['amount']))
    b = by(t['name'], summary(sum=sum(t['amount'])))

    assert a.isidentical(b)


def test_by_summary_printing():
    t = symbol('t', 'var * {name: string, amount: int32, id: int32}')
    assert str(by(t.name, total=sum(t.amount))) == \
            'by(t.name, total=sum(t.amount))'


def test_by_columns():
    t = TableSymbol('t', '{name: string, amount: int32, id: int32}')

    assert len(by(t['id'], total=t['amount'].sum()).fields) == 2
    assert len(by(t['id'], count=t['id'].count()).fields) == 2
    print(by(t, count=t.count()).fields)
    assert len(by(t, count=t.count()).fields) == 4


def test_sort():
    t = TableSymbol('t', '{name: string, amount: int32, id: int32}')
    s = t.sort('amount', ascending=True)
    print(str(s))
    assert eval(str(s)).isidentical(s)

    assert s.schema == t.schema

    assert t['amount'].sort().key == 'amount'


def test_head():
    t = TableSymbol('t', '{name: string, amount: int32, id: int32}')
    s = t.head(10)
    assert eval(str(s)).isidentical(s)

    assert s.schema == t.schema


def test_label():
    t = TableSymbol('t', '{name: string, amount: int32, id: int32}')
    quantity = (t['amount'] + 100).label('quantity')

    assert eval(str(quantity)).isidentical(quantity)

    assert quantity.fields == ['quantity']

    with pytest.raises(ValueError):
        quantity['balance']


def test_map_label():
    t = TableSymbol('t', '{name: string, amount: int32, id: int32}')
    c = t.amount.map(identity, schema='int32')
    assert c.label('bar')._name == 'bar'
    assert c.label('bar')._child.isidentical(c._child)


def test_columns():
    t = TableSymbol('t', '{name: string, amount: int32, id: int32}')
    assert list(t.fields) == ['name', 'amount', 'id']
    assert list(t['name'].fields) == ['name']
    (t['amount'] + 1).fields


def test_relabel():
    t = TableSymbol('t', '{name: string, amount: int32, id: int32}')

    rl = t.relabel({'name': 'NAME', 'id': 'ID'})
    rlc = t['amount'].relabel({'amount': 'BALANCE'})

    assert eval(str(rl)).isidentical(rl)

    print(rl.fields)
    assert rl.fields == ['NAME', 'amount', 'ID']

    assert not isscalar(rl.dshape.measure)
    assert isscalar(rlc.dshape.measure)


def test_relabel_join():
    names = TableSymbol('names', '{first: string, last: string}')

    siblings = join(names.relabel({'last': 'left'}),
                    names.relabel({'last': 'right'}), 'first')

    assert siblings.fields == ['first', 'left', 'right']


def test_map():
    t = TableSymbol('t', '{name: string, amount: int32, id: int32}')
    inc = lambda x: x + 1
    assert isscalar(t['amount'].map(inc, schema='int').dshape.measure)
    s = t['amount'].map(inc, schema='{amount: int}')
    assert not isscalar(s.dshape.measure)

    assert s.dshape == dshape('var * {amount: int}')

    expr = (t[['name', 'amount']]
            .map(identity, schema='{name: string, amount: int}'))
    assert expr._name is None


@pytest.mark.xfail(reason="Not sure that we should even support this")
def test_map_without_any_info():
    t = TableSymbol('t', '{name: string, amount: int32, id: int32}')
    assert iscolumn(t['amount'].map(inc, 'int'))
    assert not iscolumn(t[['name', 'amount']].map(identity))


def test_apply():
    t = Symbol('t', 'var * {name: string, amount: int32, id: int32}')
    s = t['amount'].apply(sum, dshape='real')
    r = t['amount'].apply(sum, dshape='3 * real')
    assert s.dshape == dshape('real')
    assert r.schema == dshape('real')


def test_TableSymbol_printing_is_legible():
    accounts = TableSymbol('accounts', '{name: string, balance: int, id: int}')

    expr = (exp(accounts.balance * 10)) + accounts['id']
    assert "exp(accounts.balance * 10)" in str(expr)
    assert "+ accounts.id" in str(expr)


def test_merge():
    t = TableSymbol('t', 'int64')
    p = TableSymbol('p', '{amount:int}')
    accounts = TableSymbol('accounts',
                           '{name: string, balance: int32, id: int32}')
    new_amount = (accounts.balance * 1.5).label('new')

    c = merge(accounts[['name', 'balance']], new_amount)
    assert c.fields == ['name', 'balance', 'new']
    assert c.schema == dshape('{name: string, balance: int32, new: float64}')

    with pytest.raises(ValueError):
        merge(t, t)
    with pytest.raises(ValueError):
        merge(t, p)


def test_merge_repeats():
    accounts = TableSymbol('accounts',
                           '{name: string, balance: int32, id: int32}')
    with pytest.raises(ValueError):
        merge(accounts, (accounts.balance + 1).label('balance'))


def test_merge_project():
    accounts = TableSymbol('accounts',
                           '{name: string, balance: int32, id: int32}')
    new_amount = (accounts['balance'] * 1.5).label('new')
    c = merge(accounts[['name', 'balance']], new_amount)

    assert c['new'].isidentical(new_amount)
    assert c['name'].isidentical(accounts['name'])

    assert c[['name', 'new']].isidentical(merge(accounts.name, new_amount))


inc = lambda x: x + 1


def test_subterms():
    a = TableSymbol('a', '{x: int, y: int, z: int}')
    assert list(a._subterms()) == [a]
    assert set(a['x']._subterms()) == set([a, a['x']])
    assert set(a['x'].map(inc, 'int')._subterms()) == \
            set([a, a['x'], a['x'].map(inc, 'int')])
    assert a in set((a['x'] + 1)._subterms())


def test_common_subexpression():
    a = TableSymbol('a', '{x: int, y: int, z: int}')

    assert common_subexpression(a).isidentical(a)
    assert common_subexpression(a, a['x']).isidentical(a)
    assert common_subexpression(a['y'] + 1, a['x']).isidentical(a)
    assert common_subexpression(a['x'].map(inc, 'int'), a['x']).isidentical(a['x'])


def test_schema_of_complex_interaction():
    a = TableSymbol('a', '{x: int, y: int, z: int}')
    expr = (a['x'] + a['y']) / a['z']
    assert expr.schema == dshape('float64')

    expr = expr.label('foo')
    assert expr.schema == dshape('float64')


def iscolumn(x):
    return isscalar(x.dshape.measure)


def test_iscolumn():
    a = TableSymbol('a', '{x: int, y: int, z: int}')
    assert not iscolumn(a)
    assert iscolumn(a['x'])
    assert not iscolumn(a[['x', 'y']])
    assert not iscolumn(a[['x']])
    assert iscolumn((a['x'] + a['y']))
    assert iscolumn(a['x'].distinct())
    assert not iscolumn(a[['x']].distinct())
    assert not iscolumn(by(a['x'], total=a['y'].sum()))
    assert iscolumn(a['x'][a['x'] > 1])
    assert not iscolumn(a[['x', 'y']][a['x'] > 1])
    assert iscolumn(a['x'].sort())
    assert not iscolumn(a[['x', 'y']].sort())
    assert iscolumn(a['x'].head())
    assert not iscolumn(a[['x', 'y']].head())

    assert iscolumn(TableSymbol('b', 'int'))
    assert not iscolumn(TableSymbol('b', '{x: int}'))


def test_discover():
    schema = '{x: int, y: int, z: int}'
    a = TableSymbol('a', schema)

    assert discover(a) == var * schema


def test_improper_selection():
    t = TableSymbol('t', '{x: int, y: int, z: int}')

    assert raises(Exception, lambda: t[t['x'] > 0][t.sort()[t['y' > 0]]])


def test_serializable():
    t = TableSymbol('t', '{id: int, name: string, amount: int}')
    import pickle
    t2 = pickle.loads(pickle.dumps(t))

    assert t.isidentical(t2)

    s = TableSymbol('t', '{id: int, city: string}')
    expr = join(t[t.amount < 0], s).sort('id').city.head()
    expr2 = pickle.loads(pickle.dumps(expr))

    assert expr.isidentical(expr2)


def test_table_coercion():
    from datetime import date
    t = TableSymbol('t', '{name: string, amount: int, timestamp: ?date}')
    assert (t.amount + '10').rhs == 10

    assert (t.timestamp < '2014-12-01').rhs == date(2014, 12, 1)


def test_isnan():
    from blaze import isnan
    t = TableSymbol('t', '{name: string, amount: real, timestamp: ?date}')

    for expr in [t.amount.isnan(), ~t.amount.isnan()]:
        assert eval(str(expr)).isidentical(expr)

    assert iscollection(t.amount.isnan().dshape)
    assert 'bool' in str(t.amount.isnan().dshape)


def test_distinct_name():
    t = TableSymbol('t', '{id: int32, name: string}')

    assert t.name.isidentical(t['name'])
    assert t.distinct().name.isidentical(t.distinct()['name'])
    assert t.id.distinct()._name == 'id'
    assert t.name._name == 'name'


def test_leaves():
    t = TableSymbol('t', '{id: int32, name: string}')
    v = TableSymbol('v', '{id: int32, city: string}')
    x = symbol('x', 'int32')

    assert t._leaves() == [t]
    assert t.id._leaves() == [t]
    assert by(t.name, count=t.id.nunique())._leaves() == [t]
    assert join(t, v)._leaves() == [t, v]
    assert join(v, t)._leaves() == [v, t]

    assert (x + 1)._leaves() == [x]


@pytest.fixture
def t():
    return TableSymbol('t', '{id: int, amount: float64, name: string}')


def funcname(x, y='<lambda>'):
    if PY3:
        return 'TestRepr.%s.<locals>.%s' % (x, y)
    return 'test_table.%s' % y


class TestRepr(object):
    def test_partial_lambda(self, t):
        expr = t.amount.map(partial(lambda x, y: x + y, 1))
        s = str(expr)
        assert s == ("Map(_child=t.amount, "
                     "func=partial(%s, 1), "
                     "_schema=None, _name0=None)" %
                     funcname('test_partial_lambda'))

    def test_lambda(self, t):
        expr = t.amount.map(lambda x: x)
        s = str(expr)
        assert s == ("Map(_child=t.amount, "
                     "func=%s, _schema=None, _name0=None)" %
                     funcname('test_lambda'))

    def test_partial(self, t):
        def myfunc(x, y):
            return x + y
        expr = t.amount.map(partial(myfunc, 1))
        s = str(expr)
        assert s == ("Map(_child=t.amount, "
                     "func=partial(%s, 1), "
                     "_schema=None, _name0=None)" % funcname('test_partial',
                                                                'myfunc'))

    def test_builtin(self, t):
        expr = t.amount.map(datetime.fromtimestamp)
        s = str(expr)
        assert s == ("Map(_child=t.amount, "
                     "func=datetime.fromtimestamp, _schema=None,"
                     " _name0=None)")

    def test_udf(self, t):
        def myfunc(x):
            return x + 1
        expr = t.amount.map(myfunc)
        s = str(expr)
        assert s == ("Map(_child=t.amount, "
                     "func=%s, _schema=None,"
                     " _name0=None)" % funcname('test_udf', 'myfunc'))

    def test_nested_partial(self, t):
        def myfunc(x, y, z):
            return x + y + z
        f = partial(partial(myfunc, 2), 1)
        expr = t.amount.map(f)
        s = str(expr)
        assert s == ("Map(_child=t.amount, func=partial(partial(%s, 2), 1),"
                     " _schema=None, _name0=None)" %
                     funcname('test_nested_partial', 'myfunc'))


def test_count_values():
    t = TableSymbol('t', '{name: string, amount: int, city: string}')
    assert t.name.count_values(sort=False).isidentical(
            by(t.name, count=t.name.count()))
    assert t.name.count_values(sort=True).isidentical(
            by(t.name, count=t.name.count()).sort('count', ascending=False))


def test_dir():
    t = TableSymbol('t', '{name: string, amount: int, dt: datetime}')
    assert 'day' in dir(t.dt)
    assert 'mean' not in dir(t.dt)
    assert 'mean' in dir(t.amount)
    assert 'like' not in dir(t[['amount', 'dt']])
    assert 'any' not in dir(t.name)


def test_distinct_column():
    t = TableSymbol('t', '{name: string, amount: int, dt: datetime}')
    assert t.name.distinct().name.dshape == t.name.distinct().dshape
    assert t.name.distinct().name.isidentical(t.name.distinct())


def test_columns_attribute_for_backwards_compatibility():
    t = TableSymbol('t', '{name: string, amount: int, dt: datetime}')

    assert t.columns == t.fields

    assert 'columns' in dir(t)
    assert 'columns' not in dir(t.name)
