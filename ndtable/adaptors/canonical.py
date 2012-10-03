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
from weakref import ref
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

# TODO: better name? Adaptor carry's too much baggage
class Adaptor(object):

    def has_op(self, op, method):
        if op == READ:
            return method & self.read_capabilities
        if op == WRITE:
            return method & self.write_capabilities

    def calculate(self, ntype):
        # Calculate the length of the buffer assuming given a
        # machine type.
        raise NotImplementedError()

class RawAdaptor(Adaptor):
    """
    Work with Python lists as if they were byte interfaces.
    """
    read_capabilities  = CONTIGIOUS | STRIDED | STREAM
    write_capabilities = CONTIGIOUS | STRIDED | STREAM

    def __init__(self, lst):
        self.lst = lst

    def calculate(self, ntype):
        # Python lists are untyped so discard information about
        # machine types.
        return len(self.lst)

    def read(self, offset, nbytes):
        return self.lst[offset:nbytes]

    def write(self, offset, nbytes):
        return

    def __repr__(self):
        return 'Raw(ptr=%r)' % id(self.lst)

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

class FileAdaptor(Adaptor):
    """
    File on local disk.
    """

    read_capabilities  = STRIDED | STREAM
    write_capabilities = STRIDED | STREAM

    def __init__(self, fname, mode='rb+'):
        self.fd = None
        self.fname = fname
        self.mode = mode
        # TODO: lazy
        self.__alloc__()

    def read(self, offset, nbytes):
        self.rd.read()

    def write(self, offset, nbytes):
        pass

    def __alloc__(self):
        self.fd = open(self.fname, self.mode)

    def __dealloc__(self):
        self.fd.close()

class SocketAdaptor(Adaptor):
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
        self.setsockopt(self.SOL_SOCKET, self.SO_REUSEADDR, 1)
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

class ImageAdaptor(Adaptor):

    def __init__(self, ifile):
        self.ifile = ifile
        self.fd = None
        self.__alloc__()

    def __alloc__(self):
        from PIL import Image
        self.fd = Image.open(self.ifile)

    def __dealloc__(self):
        self.fd.close()

    def read(self, nbytes):
        pass

    def write(self, nbytes):
        pass

class PyAdaptor(Adaptor):

    def __init__(self, pyobj):
        # Works for anything that supports memory-like access
        try:
            self.ref = memoryview(pyobj)
        except:
            self.ref = memoryview(pyobj)

    def __alloc__(self):
        pass

    def __dealloc__(self):
        pass

    def read(self, nbytes):
        pass

    def write(self, nbytes):
        pass

# TODO: probably just want to duck-type it to whatever the IO Pro
# interface uses...
class IOPro(Adaptor):
    pass
