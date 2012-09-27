"""
Toy implementations of the "canonical backends".

These are purely for internal use, are never exposed to the end-user in
any form. Adaptors simply provide a layer over the low-level IO
operations needed to produce the bytes as needed.

They also hint at the memory properties of the underlying
substrate ( i.e sockets have chunked access, but are not
seekable ).

Going up and down in the hierarchy of memory-access involves costs.
Going from contiguous to stream involves seek operations, while going
from a stream to contiguous involves copying data.

   Seek    Contiguous
     |        |         ^
     |     Strided      |
     v        |         |
           Stream      Copy
"""

import socket
from ctypes import CDLL, c_int, POINTER, byref, string_at
libc = CDLL("libc.so.6")

free   = libc.free
malloc = libc.free

CONTIGIOUS = 1
STRIDED    = 2
STREAM     = 4
CHUNKED    = 8

# TODO: ask peter
# Chunked??? For memory access that is partial but chunk sizes
# don't correlate to datashape.

READ  = 1
WRITE = 2
READWRITE = READ | WRITE

class Adaptor(object):

    def has_op(self, op, method):
        if op == READ:
            return method & self.read_capabilities
        if op == WRITE:
            return method & self.write_capabilities

class MemoryAdaptor(Adaptor):
    """
    Allocate a memory block, explicitly outside of the Python
    heap.
    """

    read_capabilities  = CONTIGIOUS | STRIDED | STREAM
    write_capabilities = CONTIGIOUS | STRIDED | STREAM

    def __init__(self):
        # TODO: lazy
        self.__alloc__()

    def read(self, offset, nbytes):
        bits = bytearray(string_at(self.block, offset))
        return memoryview(bits)

    def write(self, offset, wbytes):
        pass

    def __alloc__(self, itemsize, size):
        self.block = malloc(itemsize*size)

    def __dealloc__(self):
        self.free(self.block)

class FileAdaptor(object):
    """
    File on local disk.
    """

    read_capabilities  = STRIDED | STREAM
    write_capabilities = STRIDED | STREAM

    def __init__(self, fname, mode='r+'):
        self.fd = None
        self.mode = mode
        self.__alloc__()

    def read(self, offset, nbytes):
        self.rd.read()

    def write(self, offset, nbytes):
        pass

    def __alloc__(self):
        self.fd = open(self.fname, self.mode)

    def __dealloc__(self):
        self.fd.close()

class SocketAdaptor(object):
    """
    Stream of bytes over TCP.
    """

    read_capabilities  = STREAM
    write_capabilities = STREAM

    def __init__(self, host, port, flags=None):
        self.host = host
        self.port = port
        self.flags = flags
        self.__alloc__()

    def __alloc__(self):
        self.socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.setsockopt(self.SOL_SOCKET, self.SO_REUSEADDR,1)
        self.socket.connect((self.host, self.port))

    def __dealloc__(self):
        self.socket.close()

    def read(self, nbytes):
        msg = bytearray(nbytes)
        view = memoryview(msg)
        while view:
            rbytes = self.socket.recv_into(view, nbytes)
            yield view[rbytes:]

    def write(self, nbytes):
        pass
