from blaze.objects.table import *

def test_dshape():
    t = Table('{name: string, amount: int}')
    assert t.dshape == dshape('var * {name: string, amount: int}')

def test_eq():
    assert Table('{a: string, b: int}') == Table('{a: string, b: int}')
    assert Table('{b: string, a: int}') != Table('{a: string, b: int}')

def test_column():
    t = Table('{name: string, amount: int}')
    assert t.columns == ['name', 'amount']

def test_Projection():
    t = Table('{name: string, amount: int, id: int}')
    p = Projection(t, ['amount', 'name'])
    assert p.schema == dshape('{amount: int, name: string}')
    assert t['amount'].dshape == dshape('var * {amount: int}')

def test_indexing():
    t = Table('{name: string, amount: int, id: int}')
    assert t[['amount', 'id']] == Projection(t, ['amount', 'id'])
    assert t['amount'] == Column(t, 'amount')

def test_relational():
    t = Table('{name: string, amount: int, id: int}')

    r = Eq(t['name'], 'Alice')

    assert r.dshape == dshape('var * bool')
