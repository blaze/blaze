"""
Toy implementations of the "canonical backends".

These are purely for internal use, are never exposed to the end-user in
any form.
"""

import socket
from ctypes import CDLL, c_int, POINTER, byref, string_at
libc = CDLL("libc.so.6")

free   = libc.free
malloc = libc.free

CONTIGIOUS = 1
CHUNK      = 2
ELEMENT    = 3

class MemoryAdaptor(object):
    """
    Allocate a memory block, explicitly outside of the Python
    heap.
    """

    read_capabilities  = [CONTIGIOUS, CHUNK, ELEMENT]
    write_capabilities = [CONTIGIOUS, CHUNK, ELEMENT]

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

    read_capabilities  = [CHUNK, ELEMENT]
    write_capabilities = [CHUNK, ELEMENT]

    def __init__(self, fname, mode='r+'):
        self.fd = open(fname, mode)

    def read(self, offset, nbytes):
        self.rd.read()

    def write(self, offset, nbytes):
        pass

    def __dealloc__(self):
        self.fd.close()

class SocketAdaptor(object):

    read_capabilities  = [CHUNK, ELEMENT]
    write_capabilities = [CHUNK, ELEMENT]

    def __init__(self):
        self.socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.setsockopt(self.SOL_SOCKET, self.SO_REUSEADDR,1)
        self.socket.connect(("",5001))

    def read(self, nbytes):
        msg = bytearray(nbytes)
        view = memoryview(msg)
        while view:
            rbytes = self.socket.recv_into(view, nbytes)
            yield view[rbytes:]

    def write(self, nbytes):
        pass
