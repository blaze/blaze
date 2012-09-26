from datashape import *
from parse import parse

def test_custom_type():
    w = TypeVar('w')
    x = TypeVar('x')
    y = TypeVar('y')
    z = TypeVar('z')

    complex64 = CType('complex64')

    Quaternion = complex64*(z*y*x*w)
    RGBA = Record(R=int16, G=int16, B=int16, A=int8)

    Type.register('Quaternion', Quaternion)
    Type.register('RGBA', RGBA)

    p1 = parse('800, 600, RGBA')
    assert p1[2] is RGBA

    # We want to build records out of custom type aliases
    p2 = parse('Record(x=Quaternion, y=Quaternion)')
