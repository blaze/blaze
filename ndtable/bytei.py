# Ideas:
#  - https://wiki.continuum.io/Bytes
#  - stdlib.io
#  - carray

"""
Byte providers map a subset of the indexer space into low-level
IO operations.

The ByteProvider provides the high-level API operations on those bytes
that abstracts away the notion of whether the object is contiguous,
strided, or streamed and can extract or write bytes in any case.

If the ByteDescriptor supports an operation natively then it can perform
it in a single "instruction", if it does not then the byte interface
will devise a way to do the operation as a sequence of instructions.
"""

from byteproto import READ, WRITE, CONTIGIOUS, STRIDED, \
    CHUNKED, STREAM
from idx import Indexable

class ByteProvider(Indexable):

    def __init__(self, source):
        pass

    def __getitem__(self, indexer):
        pass

    def __getslice__(self, start, stop, step):
        pass

    def has_op(self, op, method):
        if op == READ:
            return method & self.read_capabilities
        if op == WRITE:
            return method & self.write_capabilities

    def calculate(self, ntype):
        # Calculate the length of the buffer assuming given a
        # machine type.
        raise NotImplementedError()

class ByteDescriptor(object):
    """
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
