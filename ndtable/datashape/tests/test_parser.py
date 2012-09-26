from parse import parse
from datashape import *

def test_simple_parse():
    x = parse('800, 600, RGBA')
    y = parse('Enum (1,2)')
    z = parse('300 , 400, Record(x=int64, y=int32)')

    assert type(x) is DataShape
    assert type(y) is DataShape
    assert type(z) is DataShape

    assert type(x[0]) is Integer

def test_flatten1():
    x = parse('a, ( b, ( c, ( d ) ) )')
    y = parse('a, b, c, d')

    assert x[0].symbol == 'a'
    assert x[1].symbol == 'b'
    assert x[2].symbol == 'c'
    assert x[3].symbol == 'd'

    assert y[0].symbol == 'a'
    assert y[1].symbol == 'b'
    assert y[2].symbol == 'c'
    assert y[3].symbol == 'd'
