from __future__ import absolute_import, division, print_function

import pytest
import pandas as pd
from operator import (add, sub, mul, floordiv, mod, pow, truediv, eq, ne, lt,
                      gt, le, ge, getitem)

try:
    from operator import div
except ImportError:
    from operator import truediv as div

from functools import partial
from datetime import datetime
import datashape
from blaze import CSV, Table
from blaze.expr import (TableSymbol, projection, Column, selection, ColumnWise,
                        join, cos, by, union, TableExpr, exp, distinct, Apply,
                        columnwise, eval_str, merge, common_subexpression, sum,
                        Label, ReLabel, Head, Sort, isnan, any, summary,
                        Summary, count, ScalarSymbol, like, Like)
from blaze.expr.table import _expr_child, unpack, max, min
from blaze.compatibility import PY3, _strtypes
from blaze.expr.core import discover
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

def test_table_name():
    t = TableSymbol('t', '10 * {people: string, amount: int}')
    r = TableSymbol('r', 'int64', iscolumn=True)
    with pytest.raises(ValueError):
        t.name
    with pytest.raises(ValueError):
        r.name

def test_table_symbol_bool():
    t = TableSymbol('t', '10 * {name: string, amount: int}')
    assert t.__bool__() == True


def test_nonzero():
    t = TableSymbol('t', '10 * {name: string, amount: int}')
    assert t
    assert (not not t) is True


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
    assert t.columns == ['name', 'amount']

    assert eval(str(t['name'])) == t['name']
    assert str(t['name']) == "t['name']"
    with pytest.raises(ValueError):
        t['name'].project('balance')
    with pytest.raises(ValueError):
        getitem(t, set('balance'))


def test_symbol_projection_failures():
    t = TableSymbol('t', '10 * {name: string, amount: int}')
    with pytest.raises(ValueError):
        t.project(['name', 'id'])
    with pytest.raises(ValueError):
        t.project('id')
    with pytest.raises(ValueError):
        t.project(t.dshape)


def test_Projection():
    t = TableSymbol('t', '{name: string, amount: int, id: int32}')
    p = projection(t, ['amount', 'name'])
    assert p.schema == dshape('{amount: int32, name: string}')
    print(t['amount'].dshape)
    print(dshape('var * int32'))
    assert t['amount'].dshape == dshape('var * {amount: int32}')

    assert eval(str(p)).isidentical(p)
    assert p.project(['amount','name']) == p[['amount','name']]
    with pytest.raises(ValueError):
        p.project('balance')


def test_indexing():
    t = TableSymbol('t', '{name: string, amount: int, id: int}')
    assert t[['amount', 'id']] == projection(t, ['amount', 'id'])
    assert t['amount'].isidentical(Column(t, 'amount'))


def test_relational():
    t = TableSymbol('t', '{name: string, amount: int, id: int}')

    r = (t['name'] == 'Alice')

    assert r.dshape == dshape('var * {name: bool}')


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
    assert t3

def test_path_issue():
    from blaze.api.dplyr import transform
    t = TableSymbol('t', "{ topic : string, word : string, result : ?float64}")
    t2 = transform(t, sizes=t.result.map(lambda x: (x - MIN)*10/(MAX - MIN),
                                         schema='{size: float64}'))

    assert t2.sizes in t2.children


def test_different_schema_raises():
    with tmpfile('.csv') as filename:
        df = pd.DataFrame(np.random.randn(10, 2))
        df.to_csv(filename, index=False, header=False)
        with pytest.raises(TypeError):
            Table(CSV(filename), columns=list('ab'))


def test_getattr_doesnt_override_properties():
    t = TableSymbol('t', '{iscolumn: string, schema: string}')
    assert isinstance(t.iscolumn, bool)
    assert isinstance(t.schema, DataShape)


def test_dir_contains_columns():
    t = TableSymbol('t', '{name: string, amount: int, id: int}')
    result = dir(t)
    columns_set = set(t.columns)
    assert set(result) & columns_set == columns_set


def test_selection_consistent_children():
    t = TableSymbol('t', '{name: string, amount: int, id: int}')

    expr = t['name'][t['amount'] < 0]

    assert list(expr.columns) == ['name']


def test_columnwise_syntax():
    t = TableSymbol('t', '{x: real, y: real, z: real}')
    x, y, z = t['x'], t['y'], t['z']
    assert (x + y).active_columns() == ['x', 'y']
    assert (z + y).active_columns() == ['y', 'z']
    assert ((z + y) * x).active_columns() == ['x', 'y', 'z']

    expr = (z % x * y + z ** 2 > 0) & (x < 0)
    assert isinstance(expr, ColumnWise)


def test_str():
    import re
    t = TableSymbol('t', '{name: string, amount: int, id: int}')
    expr = t[t['amount'] < 0]['name'] * 2
    assert '<class' not in str(expr)
    assert not re.search('0x[0-9a-f]+', str(expr))

    assert eval(str(expr)) == expr

    assert '*' in repr(expr)


def test_unpack():
    assert unpack('unpack') == 'unpack'


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

    assert set(j.columns) == set('wxyz')

    assert j.on_left == j.on_right == ['x', 'y']
    assert hash(j)

    assert j.columns == ['x', 'y', 'z', 'w']


def test_traverse():
    t = TableSymbol('t', '{name: string, amount: int}')
    assert t in list(t.traverse())

    expr = t[t['amount'] < 0]['name']
    trav = list(expr.traverse())
    assert any(t['amount'].isidentical(x) for x in trav)
    assert any((t['amount'] < 0).isidentical(x) for x in trav)


def test_unary_ops():
    t = TableSymbol('t', '{name: string, amount: int}')
    expr = cos(exp(t['amount']))
    assert 'cos' in str(expr)

    assert '~' in str(~(t.amount > 0))


def test_reduction():
    t = TableSymbol('t', '{name: string, amount: int32}')
    r = sum(t['amount'])
    print(type(r.dshape))
    print(type(dshape('int32')))
    print(r.dshape)
    assert r.dshape in (dshape('int32'),
                        dshape('{amount: int32}'),
                        dshape('{amount_sum: int32}'))

    assert 'amount' not in str(t.count().dshape)

    assert first(t.count().dshape[0].types)[0] in (int32, int64)

    assert 'int' in str(t.count().dshape)
    assert 'int' in str(t.nunique().dshape)
    assert 'string' in str(t['name'].max().dshape)
    assert 'string' in str(t['name'].min().dshape)
    assert 'string' not in str(t.count().dshape)

    t = TableSymbol('t', '{name: string, amount: real, id: int}')

    assert 'int' in str(t['id'].sum().dshape)
    assert 'int' not in str(t['amount'].sum().dshape)


def test_max_min_class():
    t = TableSymbol('t', '{name: string, amount: int32}')
    assert str(max(t).dtype) == '{ name : string, amount : int32 }'
    assert str(min(t).dtype) == '{ name : string, amount : int32 }'


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
    assert t_ds.columns is not None

    t_sch = TableSymbol('t', dshape=ds.subshape[0])
    assert t_sch.columns is not None

    assert t_ds.isidentical(t_sch)


class TestScalarArithmetic(object):
    ops = {'+': add, '-': sub, '*': mul, '/': truediv, '//': floordiv, '%': mod,
           '**': pow, '==': eq, '!=': ne, '<': lt, '>': gt, '<=': le, '>=': ge}

    def test_scalar_arith(self, symsum):
        def runner(f):
            result = f(r, 1)
            assert eval('r %s 1' % op).isidentical(result)

            result = f(r, r)
            assert eval('r %s r' % op).isidentical(result)

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
    assert s.dshape == dshape('{num: int32, total: int32}')
    assert hash(s)
    assert eval(str(s)).isidentical(s)

    assert 'summary(' in str(s)
    assert 'total=' in str(s)
    assert 'num=' in str(s)
    assert str(t.amount.sum()) in str(s)

    assert not summary(total=t.amount.sum()).child.isidentical(
            t.amount.sum())
    assert isinstance(summary(total=t.amount.sum() + 1).child, TableExpr)


def test_reduction_arithmetic():
    t = TableSymbol('t', '{id: int32, name: string, amount: int32}')
    expr = t.amount.sum() + 1
    assert eval(str(expr)).isidentical(expr)


def test_Distinct():
    t = TableSymbol('t', '{name: string, amount: int32}')
    r = distinct(t['name'])
    print(r.dshape)
    assert r.dshape  == dshape('var * {name: string}')

    r = t.distinct()
    assert r.dshape  == t.dshape


def test_by():
    t = TableSymbol('t', '{name: string, amount: int32, id: int32}')
    r = by(t['name'], sum(t['amount']))

    print(r.schema)
    assert isinstance(r.schema[0], Record)
    assert str(r.schema[0]['name']) == 'string'


def test_by_summary():
    t = TableSymbol('t', '{name: string, amount: int32, id: int32}')
    a = by(t['name'], sum=sum(t['amount']))
    b = by(t['name'], summary(sum=sum(t['amount'])))

    assert a.isidentical(b)


def test_by_columns():
    t = TableSymbol('t', '{name: string, amount: int32, id: int32}')

    assert len(by(t['id'], t['amount'].sum()).columns) == 2
    assert len(by(t['id'], t['id'].count()).columns) == 2
    print(by(t, t.count()).columns)
    assert len(by(t, t.count()).columns) == 4


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

    assert quantity.columns == ['quantity']

    with pytest.raises(ValueError):
        quantity.project('balance')


def test_map_label():
    t = TableSymbol('t', '{name: string, amount: int32, id: int32}')
    c = t.amount.map(identity, schema='{foo: int32}')
    assert c.label('bar').name == 'bar'
    assert c.label('bar').child.isidentical(c.child)



def test_columns():
    t = TableSymbol('t', '{name: string, amount: int32, id: int32}')
    assert list(t.columns) == ['name', 'amount', 'id']
    assert list(t['name'].columns) == ['name']
    (t['amount'] + 1).columns


def test_relabel():
    t = TableSymbol('t', '{name: string, amount: int32, id: int32}')

    rl = t.relabel({'name': 'NAME', 'id': 'ID'})
    rlc = t['amount'].relabel({'amount': 'BALANCE'})

    assert eval(str(rl)).isidentical(rl)

    print(rl.columns)
    assert rl.columns == ['NAME', 'amount', 'ID']

    assert not rl.iscolumn
    assert rlc.iscolumn


def test_relabel_join():
    names = TableSymbol('names', '{first: string, last: string}')

    siblings = join(names.relabel({'last': 'left'}),
                    names.relabel({'last': 'right'}), 'first')

    assert siblings.columns == ['first', 'left', 'right']


def test_map():
    t = TableSymbol('t', '{name: string, amount: int32, id: int32}')
    r = TableSymbol('s', 'int64')
    inc = lambda x: x + 1
    assert t['amount'].map(inc).iscolumn
    assert t['amount'].map(inc, schema='{amount: int}').iscolumn
    s = t['amount'].map(inc, schema='{amount: int}', iscolumn=False)
    assert not s.iscolumn

    assert s.dshape == dshape('var * {amount: int}')

    assert not t[['name', 'amount']].map(identity).iscolumn

    with pytest.raises(ValueError):
        t[['name', 'amount']].map(identity, schema='{name: string, amount: int}').name


def test_apply():
    t = TableSymbol('t', '{name: string, amount: int32, id: int32}')
    s = Apply(sum, t['amount'], dshape='real')
    r = Apply(sum, t['amount'], dshape='3 * real')
    l = Apply(sum, t['amount'])
    assert s.dshape == dshape('real')
    assert r.schema == dshape("float64")
    
    with pytest.raises(TypeError):
        s.schema
    with pytest.raises(NotImplementedError):
        l.dshape


def test_columnwise():
    from blaze.expr.scalar import Add, Eq, Mult, Le
    t = TableSymbol('t', '{x: int, y: int, z: int}')
    t2 = TableSymbol('t', '{a: int, b: int, c: int}')
    x = t['x']
    y = t['y']
    z = t['z']
    a = t2['a']
    b = t2['b']
    c = t2['c']

    assert str(columnwise(Add, x, y).expr) == 'x + y'
    assert columnwise(Add, x, y).child.isidentical(t)

    c1 = columnwise(Add, x, y)
    c2 = columnwise(Mult, x, z)

    assert eval_str(columnwise(Eq, c1, c2).expr) == '(x + y) == (x * z)'
    assert columnwise(Eq, c1, c2).child.isidentical(t)

    assert str(columnwise(Add, x, 1).expr) == 'x + 1'

    assert str(x <= y) == "t['x'] <= t['y']"
    assert str(x >= y) == "t['x'] >= t['y']"
    assert str(x | y) == "t['x'] | t['y']"
    assert str(x.__ror__(y)) == "t['y'] | t['x']"
    assert str(x.__rand__(y)) == "t['y'] & t['x']"

    with pytest.raises(ValueError):
        columnwise(Add, x, a)


def test_expr_child():
    t = TableSymbol('t', '{x: int, y: int, z: int}')
    w = t['x'].label('w')
    assert str(_expr_child(w)) == '(x, t)'


def test_TableSymbol_printing_is_legible():
    accounts = TableSymbol('accounts', '{name: string, balance: int, id: int}')

    expr = (exp(accounts['balance'] * 10)) + accounts['id']
    assert "exp(accounts['balance'] * 10)" in str(expr)
    assert "+ accounts['id']" in str(expr)


def test_dtype():
    accounts = TableSymbol('accounts',
                           '{name: string, balance: int32, id: int32}')

    assert accounts['name'].dtype == dshape('string')
    assert accounts['balance'].dtype == dshape('int32')
    assert (accounts['balance'] > accounts['id']).dtype == dshape('bool')


def test_merge():
    t = TableSymbol('t', 'int64')
    p = TableSymbol('p', '{amount:int}')
    accounts = TableSymbol('accounts',
                           '{name: string, balance: int32, id: int32}')
    new_amount = (accounts['balance'] * 1.5).label('new')

    c = merge(accounts[['name', 'balance']], new_amount)
    assert c.columns == ['name', 'balance', 'new']

    with pytest.raises(TypeError):
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
    assert list(a.subterms()) == [a]
    assert set(a['x'].subterms()) == set([a, a['x']])
    assert set(a['x'].map(inc).subterms()) == set([a, a['x'], a['x'].map(inc)])
    assert a in set((a['x'] + 1).subterms())


def test_common_subexpression():
    a = TableSymbol('a', '{x: int, y: int, z: int}')

    assert common_subexpression(a).isidentical(a)
    assert common_subexpression(a, a['x']).isidentical(a)
    assert common_subexpression(a['y'] + 1, a['x']).isidentical(a)
    assert common_subexpression(a['x'].map(inc), a['x']).isidentical(a['x'])


def test_schema_of_complex_interaction():
    a = TableSymbol('a', '{x: int, y: int, z: int}')
    expr = (a['x'] + a['y']) / a['z']
    assert expr.dtype == dshape('real')

    expr = expr.label('foo')
    print(expr.dtype)
    assert expr.dtype == dshape('real')


def test_iscolumn():
    a = TableSymbol('a', '{x: int, y: int, z: int}')
    assert not a.iscolumn
    assert a['x'].iscolumn
    assert not a[['x', 'y']].iscolumn
    assert not a[['x']].iscolumn
    assert (a['x'] + a['y']).iscolumn
    assert a['x'].distinct().iscolumn
    assert not a[['x']].distinct().iscolumn
    assert not by(a['x'], a['y'].sum()).iscolumn
    assert a['x'][a['x'] > 1].iscolumn
    assert not a[['x', 'y']][a['x'] > 1].iscolumn
    assert a['x'].sort().iscolumn
    assert not a[['x', 'y']].sort().iscolumn
    assert a['x'].head().iscolumn
    assert not a[['x', 'y']].head().iscolumn

    assert TableSymbol('b', '{x: int}', iscolumn=True).iscolumn
    assert not TableSymbol('b', '{x: int}', iscolumn=False).iscolumn
    assert not TableSymbol('b', '{x: int}', iscolumn=True).isidentical(
            TableSymbol('b', '{x: int}', iscolumn=False))


def test_discover():
    schema = '{x: int, y: int, z: int}'
    a = TableSymbol('a', schema)

    assert discover(a) == var * schema


def test_improper_selection():
    t = TableSymbol('t', '{x: int, y: int, z: int}')

    assert raises(Exception, lambda: t[t['x'] > 0][t.sort()[t['y' > 0]]])


def test_union():
    schema = '{x: int, y: int, z: int}'
    a = TableSymbol('a', schema)
    b = TableSymbol('b', schema)
    c = TableSymbol('c', schema)

    u = union(a, b, c)
    assert u.schema == a.schema

    assert raises(Exception,
                  lambda: union(a, TableSymbol('q', '{name: string}')))


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
    assert (t.amount + '10').expr.rhs == 10

    assert (t.timestamp < '2014-12-01').expr.rhs == date(2014, 12, 1)


def test_isnan():
    t = TableSymbol('t', '{name: string, amount: int, timestamp: ?date}')

    for expr in [t.amount.isnan(), ~t.amount.isnan()]:
        assert eval(str(expr)).isidentical(expr)

    assert isinstance(t.amount.isnan(), TableExpr)
    assert 'bool' in str(t.amount.isnan().dshape)


def test_columnwise_naming():
    t = TableSymbol('t', '{x: int, y: int, z: int}')

    assert t.x.name == 'x'
    assert (t.x + 1).name == 'x'


def test_scalar_expr():
    t = TableSymbol('t', '{x: int64, y: int32, z: int64}')
    x = t.x.expr
    y = t.y.expr
    assert 'int64' in str(x.dshape)
    assert 'int32' in str(y.dshape)

    expr = (t.x + 1).expr
    assert expr.inputs[0].dshape == x.dshape
    assert expr.inputs[0].isidentical(x)

    t = TableSymbol('t', '{ amount : int64, id : int64, name : string }')
    expr = (t.amount + 1).expr
    assert 'int64' in str(expr.inputs[0].dshape)


def test_distinct_name():
    t = TableSymbol('t', '{id: int32, name: string}')

    assert t.name.isidentical(t['name'])
    assert t.distinct().name.isidentical(t.distinct()['name'])
    assert t.id.distinct().name == 'id'
    assert t.name.name == 'name'


def test_leaves():
    t = TableSymbol('t', '{id: int32, name: string}')
    v = TableSymbol('v', '{id: int32, city: string}')
    x = ScalarSymbol('x', 'int32')

    assert t.leaves() == [t]
    assert t.id.leaves() == [t]
    assert by(t.name, t.id.nunique()).leaves() == [t]
    assert join(t, v).leaves() == [t, v]
    assert join(v, t).leaves() == [v, t]

    assert (x + 1).leaves() == [x]


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
        assert s == ("Map(child=t['amount'], "
                     "func=partial(%s, 1), "
                     "_schema=None, _iscolumn=None)" %
                     funcname('test_partial_lambda'))

    def test_lambda(self, t):
        expr = t.amount.map(lambda x: x)
        s = str(expr)
        assert s == ("Map(child=t['amount'], "
                     "func=%s, _schema=None, _iscolumn=None)" %
                     funcname('test_lambda'))

    def test_partial(self, t):
        def myfunc(x, y):
            return x + y
        expr = t.amount.map(partial(myfunc, 1))
        s = str(expr)
        assert s == ("Map(child=t['amount'], "
                     "func=partial(%s, 1), "
                     "_schema=None, _iscolumn=None)" % funcname('test_partial',
                                                                'myfunc'))

    def test_builtin(self, t):
        expr = t.amount.map(datetime.fromtimestamp)
        s = str(expr)
        assert s == ("Map(child=t['amount'], "
                     "func=datetime.fromtimestamp, _schema=None,"
                     " _iscolumn=None)")

    def test_udf(self, t):
        def myfunc(x):
            return x + 1
        expr = t.amount.map(myfunc)
        s = str(expr)
        assert s == ("Map(child=t['amount'], "
                     "func=%s, _schema=None,"
                     " _iscolumn=None)" % funcname('test_udf', 'myfunc'))

    def test_nested_partial(self, t):
        def myfunc(x, y, z):
            return x + y + z
        f = partial(partial(myfunc, 2), 1)
        expr = t.amount.map(f)
        s = str(expr)
        assert s == ("Map(child=t['amount'], func=partial(partial(%s, 2), 1),"
                     " _schema=None, _iscolumn=None)" %
                     funcname('test_nested_partial', 'myfunc'))


def test_like():
    t = TableSymbol('t', '{name: string, amount: int, city: string}')

    expr = like(t, name='Alice*')

    assert eval(str(expr)).isidentical(expr)
    assert expr.schema == t.schema
    assert expr.dshape[0] == datashape.var


def test_count_values():
    t = TableSymbol('t', '{name: string, amount: int, city: string}')
    assert t.name.count_values(sort=False).isidentical(
            by(t.name, count=t.name.count()))
    assert t.name.count_values(sort=True).isidentical(
            by(t.name, count=t.name.count()).sort('count', ascending=False))
