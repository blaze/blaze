# Ideas:
#  - https://wiki.continuum.io/Bytes
#  - stdlib.io
#  - carray

import struct
from ctypes import Structure, c_void_p, c_int
from sources.canonical import READ, WRITE, CONTIGIOUS, STRIDED, \
    CHUNKED, STREAM
from idx import Indexable

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

class Flags:
    ACCESS_DEFAULT = 1
    ACCESS_READ    = 2
    ACCESS_WRITE   = 3
    ACCESS_COPY    = 4 # Copied locally, but not committed
    ACCESS_APPEND  = 5

class ByteProvider(Indexable):

    def __init__(self, source):
        pass

    def __getitem__(self, indexer):
        pass

    def __getslice__(self, start, stop, step):
        pass

class ByteDescriptor(object):
    """
    Source provides low-level IO operations. The byte interface provides
    the high-level API operations on those bytes that abstracts away the
    notion of whether the object is contiguous, strided, or streamed and
    can extract or write bytes in any case.

    If the source supports an operation natively then it can perform it
    in a single "instruction", if it does not then the byte interface
    will devise a way to do the operation as a sequence of instructions.
    """

    def __init__(self, source, flags):
        self.source = source
        self.flags = flags

    def close(self):
        pass

    def flush(self):
        pass

    def move(self, dest, src, count):
        pass

    def read(self, num):
        pass

    # Size of the bytes
    def write(self, n):
        pass

    # Contiguous Operations
    # ---------------------
    def seek(self, offset):
        pass

    def peek(self, n):
        # Return bytes from the stream without advancing the position
        pass

    def tell(self):
        pass

    # Read element at cursor
    def read_slice(self, start, stop):
        pass

    # Write element at cursor
    def write_slice(self, start, stop):
        pass

    __next__ = read_slice
