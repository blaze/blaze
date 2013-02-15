from blaze.datashape import *
from blaze.datashape.parser import parse
from blaze.datashape.record import RecordDecl, derived
from blaze.datashape.coretypes import _reduce

from unittest import skip

def test_simple_parse():
    x = parse('2, 3, int32')
    y = parse('300 , 400, {x: int64; y: int32}')

    assert type(x) == DataShape
    assert type(y) == DataShape

    assert type(y[0]) == Fixed
    assert type(y[1]) == Fixed
    assert type(y[2]) == Record

    rec = y[2]

    assert rec['x'] == int64
    assert rec['y'] == int32

def test_compound_record1():
    p = parse('6, {x:int; y:float; z:str}')

    assert type(p[0]) == Fixed
    assert type(p[1]) == Record

def test_compound_record2():
    p = parse('{ a: { x: int; y: int }; b: {w: int; u: int } }')

    assert type(p[0]) == Record

def test_free_variables():
    p = parse('N, M, 800, 600, int32')

    assert type(p[0]) == TypeVar
    assert type(p[1]) == TypeVar
    assert type(p[2]) == Fixed
    assert type(p[3]) == Fixed
    assert type(p[4]) == CType

def test_parse_equality():
    x = parse('800, 600, int64')
    y = parse('800, 600, int64')

    assert x._equal(y)

def test_parse_vars():
    x = parse('Range(1,2)')

    assert x[0].lower == 1
    assert x[0].upper == 2

def test_parse_either():
    x = parse('Either(int64, float64)')
    assert x[0].a == int64
    assert x[0].b == float64

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
