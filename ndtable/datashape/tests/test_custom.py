from datashape import *
from parse import parse

w = TypeVar('w')
x = TypeVar('x')
y = TypeVar('y')
z = TypeVar('z')

Quaternion = complex64*(z*y*x*w)
RGBA = Record(R=int16, G=int16, B=int16, A=int8)

def setUp():
    Type.register('Quaternion', Quaternion)
    Type.register('RGBA', RGBA)

def test_custom_type():
    p1 = parse('800, 600, RGBA')
    assert p1[2] is RGBA

    # We want to build records out of custom type aliases
    p2 = parse('Record(x=Quaternion, y=Quaternion)')

def test_custom_stream():
    p1 = parse('Stream( RGBA )')
