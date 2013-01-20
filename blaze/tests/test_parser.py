from blaze.datashape import *
from blaze.datashape.parse import parse
from blaze.datashape.record import RecordDecl, derived
from blaze.datashape.coretypes import _reduce

from unittest import skip

def test_simple_parse():
    x = parse('2, 3, int32')
    y = parse('300 , 400, {x: int64, y: int32}')

    assert type(x) is DataShape
    assert type(y) is DataShape

    assert type(y[0]) is Fixed
    assert type(y[1]) is Fixed
    assert type(y[2]) is Record

    rec = y[2]

    assert rec['x'] is int64
    assert rec['y'] is int32

def test_compound_record1():
    p = parse('6, {x:int, y:float, z:str}')

    assert type(p[0]) is Fixed
    assert type(p[1]) is Record

def test_compound_record2():
    p = parse('{ a: { x: int, y: int }, b: {w: int, u: int } }')

    assert type(p[0]) is Record

def test_free_variables():
    p = parse('N, M, 800, 600, int32')

    assert type(p[0]) is TypeVar
    assert type(p[1]) is TypeVar
    assert type(p[2]) is Fixed
    assert type(p[3]) is Fixed
    assert type(p[4]) is CType

# TODO: INVALID
def test_flat_datashape():
    p = parse('N, M, 800, 600, (int16, int16, int16, int8)')

    assert type(p[0]) is TypeVar
    assert type(p[1]) is TypeVar
    assert type(p[2]) is Fixed
    assert type(p[3]) is Fixed

    assert p[4:8] == (int16, int16, int16, int8)

def test_flatten1():
    x = parse('a, ( b, ( c, ( d ) ) )')
    y = parse('a, b, c, d')

    assert len(x.parameters) == len(y.parameters)

    assert x[0].symbol == 'a'
    assert x[1].symbol == 'b'
    assert x[2].symbol == 'c'
    assert x[3].symbol == 'd'

    assert y[0].symbol == 'a'
    assert y[1].symbol == 'b'
    assert y[2].symbol == 'c'
    assert y[3].symbol == 'd'

    assert _reduce(x) == _reduce(y)

def test_flatten2():
    x = parse('a, ( b, ( c, d ) )')
    y = parse('a, b, c, d')

    assert len(x.parameters) == len(y.parameters)

    assert x[0].symbol == 'a'
    assert x[1].symbol == 'b'
    assert x[2].symbol == 'c'
    assert x[3].symbol == 'd'

    assert y[0].symbol == 'a'
    assert y[1].symbol == 'b'
    assert y[2].symbol == 'c'
    assert y[3].symbol == 'd'

    assert _reduce(x) == _reduce(y)

def test_parse_equality():
    x = parse('800, 600, int64')
    y = parse('800, 600, int64')

    assert x._equal(y)

def test_parse_fixed_integer_diff():
    x = parse('1, int32')
    y = parse('{1}, int32')

    assert type(x[0]) is Fixed
    assert type(y[0][0]) is Integer

def test_parse_ctypes():
    x = parse('800, 600, double')
    y = parse('800, 600, PyObject')

def test_parse_vars():
    x = parse('Range(1,2)')

    assert x[0].lower == 1
    assert x[0].upper == 2

def test_parse_na():
    x = parse('NA')
    assert x[0] is na

def test_parse_either():
    x = parse('Either(int64, NA)')
    assert x[0].a == int64
    assert x[0].b is na

def test_parse_blob_varchar():
    p1 = parse('2, 3, Varchar(5)')
    p2 = parse('2, 3, blob')

    assert type(p1[2]) is Varchar
    assert type(p2[2]) is Blob

    # Deconstructing the type
    assert p1[2].maxlen == 5

def test_parse_string():
    p1 = parse('2, 3, String(5)')

    assert type(p1[2]) is String

    # Deconstructing the type
    assert p1[2].fixlen == 5

def test_custom_record():

    class Stock1(RecordDecl):
        name   = string
        open   = float_
        close  = float_
        max    = int64
        min    = int64
        volume = float_

        @derived('int64')
        def mid(self):
            return (self.min + self.max)/2

    assert Stock1.mid

@skip
def test_custom_record_infer():

    class Stock2(RecordDecl):
        name   = string
        open   = float_
        close  = float_
        max    = int64
        min    = int64
        volume = float_

        @derived()
        def mid(self):
            return (self.min + self.max)/2

    assert Stock2.mid
