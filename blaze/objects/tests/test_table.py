from blaze.objects.table import *

def test_dshape():
    t = Table('{name: string, amount: int}')
    assert t.dshape == dshape('var * {name: string, amount: int}')

