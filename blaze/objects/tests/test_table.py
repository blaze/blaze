from blaze.objects.table import *

def test_dshape():
    t = Table('{name: string, amount: int}')
    assert t.dshape == dshape('var * {name: string, amount: int}')

def test_column():
    t = Table('{name: string, amount: int}')
    assert t.columns == ['name', 'amount']

