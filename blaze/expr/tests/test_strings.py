import datashape
from blaze.expr import TableSymbol, like, Like

def test_like():
    t = TableSymbol('t', '{name: string, amount: int, city: string}')

    expr = like(t, name='Alice*')

    assert eval(str(expr)).isidentical(expr)
    assert expr.schema == t.schema
    assert expr.dshape[0] == datashape.var

