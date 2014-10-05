from blaze.expr.predicates import *
from blaze.expr import TableSymbol, Symbol

t = TableSymbol('t', '{name: string, amount: int}')
s = Symbol('s', 'string')

def test_istabular():
    assert istabular(t)
    assert not istabular(s)

def test_iscolumn():
    assert not iscolumn(t)
    assert iscolumn(t.name)

