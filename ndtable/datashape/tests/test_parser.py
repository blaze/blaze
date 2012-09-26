from parse import parse
from datashape import *

def test_simple_parse():
    x = parse('800, 600, RGBA')
    y = parse('Enum (1,2)')
    z = parse('300 , 400, Record(x=int64, y=int32)')

    assert type(x) is DataShape
    assert type(y) is DataShape
    assert type(z) is DataShape
