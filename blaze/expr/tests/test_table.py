from __future__ import absolute_import, division, print_function

from blaze.expr.table import *
from datashape import dshape


def test_dshape():
    t = TableExpr('{name: string, amount: int}')
    assert t.dshape == dshape('var * {name: string, amount: int}')


def test_eq():
    assert TableExpr('{a: string, b: int}') == TableExpr('{a: string, b: int}')
    assert TableExpr('{b: string, a: int}') != TableExpr('{a: string, b: int}')


def test_column():
    t = TableExpr('{name: string, amount: int}')
    assert t.columns == ['name', 'amount']


def test_Projection():
    t = TableExpr('{name: string, amount: int, id: int}')
    p = Projection(t, ['amount', 'name'])
    assert p.schema == dshape('{amount: int, name: string}')
    assert t['amount'].dshape == dshape('var * {amount: int}')


def test_indexing():
    t = TableExpr('{name: string, amount: int, id: int}')
    assert t[['amount', 'id']] == Projection(t, ['amount', 'id'])
    assert t['amount'] == Column(t, 'amount')


def test_relational():
    t = TableExpr('{name: string, amount: int, id: int}')

    r = Eq(t['name'], 'Alice')

    assert r.dshape == dshape('var * bool')


def test_selection():
    t = TableExpr('{name: string, amount: int, id: int}')

    s = Selection(t, Eq(t['name'], 'Alice'))

    assert s.dshape == t.dshape


def test_selection_by_indexing():
    t = TableExpr('{name: string, amount: int, id: int}')

    result = t[t['name'] == 'Alice']
    expected = Selection(t, Eq(Column(t, 'name'), 'Alice'))
    assert result == expected


def test_columnwise():
    t = TableExpr('{x: real, y: real, z: real}')
    x, y, z = t['x'], t['y'], t['z']
    expr = z % x * y + z ** 2
    assert isinstance(expr, Column)


def test_str():
    import re
    t = TableExpr('{name: string, amount: int, id: int}')
    expr = t[['name', 'id']][t['amount'] < 0]
    print(str(expr))
    assert '<class' not in str(expr)
    assert not re.search('0x[0-9a-f]+', str(expr))

    # assert eval(str(expr)) == expr

def test_join():
    t = TableExpr('{name: string, amount: int}')
    s = TableExpr('{name: string, id: int}')
    j = Join(t, s, 'name', 'name')

    assert j.schema == dshape('{name: string, amount: int, id: int}')

    assert Join(t, s, 'name') == Join(t, s, 'name')
