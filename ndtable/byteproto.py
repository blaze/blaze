import struct
from ctypes import Structure, c_void_p, c_int

CONTIGIOUS = 1
STRIDED    = 2
STREAM     = 4
CHUNKED    = 8

READ  = 1
WRITE = 2
READWRITE = READ | WRITE

class Flags:
    ACCESS_DEFAULT = 1
    ACCESS_READ    = 2
    ACCESS_WRITE   = 3
    ACCESS_COPY    = 4
    ACCESS_APPEND  = 5

class Buffer(Structure):
    _fields_ = [
        ('data'     , c_void_p) ,
        ('offset'   , c_int)    ,
        ('stride'   , c_int*2)  ,
        ('itemsize' , c_int)    ,
        ('flags'    , c_int)    ,
    ]

def BufferList(n):
    class List(Structure):
        _fields_ = [
            ('buffers', Buffer*n)
        ]
        def __iter__(self):
            for b in self.buffers:
                yield b

    return List

def StreamList(n):
    class Stream(Structure):
        _fields_ = [
            ('index', c_int),
            ('next' , c_void_p)
        ]
        def __iter__(self):
            self.index = 0
            return self

        def __next__(self):
            self.index += 1
            yield self.next

    return Stream
