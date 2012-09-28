# Ideas:
#  - https://wiki.continuum.io/Bytes
#  - stdlib.io
#  - carray

from adaptors.canonical import READ, WRITE, CONTIGIOUS, STRIDED, \
    CHUNKED, STREAM

class Flags:
    ACCESS_DEFAULT = 1
    ACCESS_READ    = 2
    ACCESS_WRITE   = 3
    ACCESS_COPY    = 4 # Copied locally, but not comitted
    ACCESS_APPEND  = 5

class Bytes(object):

    def __init__(self, substrate=None):
        pass

    def as_contigious(self, start, stop):
        pass

    def as_strided(self):
        pass

    def as_stream(self, count):
        pass

class ByteI(object):
    """
    Adaptor provides low-level IO operations. The byte interface
    provides the high-level API operations on those bytes that abstracts
    away the notion of whether the object is contiguous, strided, or
    streamed and can extract or write bytes in any case.

    If the adaptor supports an operation natively then it can perform it
    in a single "instruction", if it does not then the byte interface
    will devise a way to do the operation as a sequence of instructions.
    """

    def __init__(self, adaptor, flags):
        self.adaptor = adaptor
        self.flags = flags

    def close(self):
        pass

    def flush(self):
        pass

    def move(self, dest, src, count):
        pass

    def read(self, num):
        if self.adaptor.has_op(READ, CONTIGIOUS):
            pass
        elif self.adaptor.has_op(READ, CHUNKED):
            pass
        elif self.adaptor.has_op(READ, STREAM):
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
