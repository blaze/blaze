from blaze.expr.predicates import *
from blaze.expr import TableSymbol, ScalarSymbol

t = TableSymbol('t', '{name: string, amount: int}')
s = ScalarSymbol('s', 'string')

def test_istabular():
    assert istabular(t)
    assert not istabular(s)

def test_isscalar():
    assert not isscalar(t)
    assert isscalar(s)

def test_iscolumn():
    assert not iscolumn(t)
    assert iscolumn(t.name)

