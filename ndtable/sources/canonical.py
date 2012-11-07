"""
Toy implementations of the "canonical backends".

These are purely for internal use, are never exposed to the end-user in
any form. Adaptors simply provide a layer over the low-level IO
operations needed to produce the bytes as needed.

They also hint at the memory properties of the underlying
substrate ( i.e sockets have chunked access, but are not
seekable ).
"""

import numpy as np

import socket

from ndtable.datashape import Fixed, pyobj
from ndtable.datashape.coretypes import CType, from_numpy
from ndtable.byteproto import CONTIGUOUS, STRIDED, STREAM
from ndtable.byteprovider import ByteProvider

from carray import carray

#------------------------------------------------------------------------
# Numpy Compat Layer
#------------------------------------------------------------------------

class ArraySource(ByteProvider):
    """
    The buffer from a Numpy array used a data source

    Only used for bootstrapping. Will be removed later.
    """

    read_capabilities  = CONTIGUOUS | STRIDED | STREAM
    write_capabilities = CONTIGUOUS | STRIDED | STREAM

    def __init__(self, lst):
        self.na = np.array(lst)

    def calculate(self, ntype):
        return from_numpy(self.na.shape, self.na.dtype)

    @classmethod
    def empty(self, datashape):
        shape, dtype = from_numpy(datashape)
        return ArraySource(np.ndarray(shape, dtype))

    def __repr__(self):
        return 'Numpy(ptr=%r, dtype=%s, shape=%r)' % (
            id(self.na),
            self.na.dtype,
            self.na.shape,
        )

#------------------------------------------------------------------------
# "CArray" byte provider
#------------------------------------------------------------------------

# The canonical backend

class CArraySource(ByteProvider):
    read_capabilities  = CONTIGUOUS
    write_capabilities = CONTIGUOUS

    def __init__(self, ca):
        """
        CArray object passed directly into the constructor,
        ostensibly this is just a thin wrapper that consumes a
        reference.
        """
        self.ca = ca

    @classmethod
    def empty(self, datashape):
        """
        Create a CArraySource from a datashape specification,
        downcasts into Numpy dtype and shape tuples if possible
        otherwise raises an exception.
        """
        shape, dtype = from_numpy(datashape)
        return CArraySource(carray([], dtype))

    def read(self, offset, nbytes):
        """
        The future read interface for the CArray.
        """
        raise NotImplementedError

    def __repr__(self):
        return 'CArray(ptr=%r)' % id(self.ca)

class PythonSource(ByteProvider):
    """
    Work with a immutable Python list as if it were byte interfaces.
    """
    read_capabilities  = CONTIGUOUS | STRIDED | STREAM
    write_capabilities = CONTIGUOUS | STRIDED | STREAM

    def __init__(self, lst):
        assert isinstance(lst, list)
        self.lst = lst

    def read(self, offset, nbytes):
        return self.lst[offset:nbytes]

    def __repr__(self):
        return 'PyObject(ptr=%r)' % id(self.lst)

class ByteSource(ByteProvider):
    """
    A raw block of memory layed out in raditional NumPy fashion.
    """

    read_capabilities  = CONTIGUOUS | STRIDED | STREAM
    write_capabilities = CONTIGUOUS | STRIDED | STREAM

    def __init__(self, bits):
        # TODO: lazy
        self.bits = bytearray(bits)
        self.mv   = memoryview(self.bits)

    def read(self, offset, nbytes):
        return self.mv[offset: offset+nbytes]

    def write(self, offset, wbytes):
        raise NotImplementedError

    def __repr__(self):
        return 'Bytes(%s...)' % self.bits[0:10]

#------------------------------------------------------------------------
# Non Memory Sources
#------------------------------------------------------------------------

class FileSource(ByteProvider):
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

    def __repr__(self):
        return 'File(fileno=%s, %s)' % (self.fd.fileno, self.fname)

class SocketSource(ByteProvider):
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

    def __repr__(self):
        return 'Socket(fileno=%s)' % (self.fd.fileno)

class ImageSource(ByteProvider):

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
