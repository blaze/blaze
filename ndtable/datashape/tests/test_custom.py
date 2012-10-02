from ndtable.datashape import *

w = TypeVar('w')
x = TypeVar('x')
y = TypeVar('y')
z = TypeVar('z')

n = TypeVar('n')

Quaternion = complex64*(z*y*x*w)
RGBA = Record(R=int16, G=int16, B=int16, A=int8)
File = string*n

def setUp():
    Type.register('Quaternion', Quaternion)
    Type.register('RGBA', RGBA)
    Type.register('File', File)

def test_custom_type():
    p1 = datashape('800, 600, RGBA')
    assert p1[2] is RGBA

    # We want to build records out of custom type aliases
    p2 = datashape('Record(x=Quaternion, y=Quaternion)')

def test_custom_stream():
    p1 = datashape('Stream, RGBA')

def test_custom_csv_like():
    # A csv-like file is a variable-length strings
    p1 = datashape('n, string')
    p2 = datashape('File')
    assert p1 == p2
