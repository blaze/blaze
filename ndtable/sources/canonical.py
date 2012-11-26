"""
Toy implementations of the byte backends, the substrate that
abstracts between raw bytes and data points.
"""

import numpy as np

import socket

from ndtable import carray

from ndtable.datashape import Fixed, dynamic, string, pyobj
from ndtable.datashape.coretypes import from_numpy
from ndtable.byteproto import CONTIGUOUS, CHUNKED, STREAM
from ndtable.byteprovider import ByteProvider

#------------------------------------------------------------------------
# Chunked Array
#------------------------------------------------------------------------

class CArraySource(ByteProvider):
    """ Chunked array is the default storage engine for Blaze arrays
    when no layout is specified. """

    read_capabilities  = CHUNKED
    write_capabilities = CHUNKED

    def __init__(self, ca):
        """ CArray object passed directly into the constructor,
        ostensibly this is just a thin wrapper that consumes a
        reference.
        """
        self.ca = carray.carray(ca)

    @property
    def nchunks(self):
        return self.ca.nchunks

    @property
    def partitions(self):
        """
        Return the partitions of elemenets in the array. The data
        bounds of each chunk.
        """
        return self.ca.partitions

    @staticmethod
    def infer_datashape(source):
        """
        The user has only provided us with a Python object (
        could be a buffer interface, a string, a list, list of
        lists, etc) try our best to infer what the datashape
        should be in the context of this datasource.
        """
        if isinstance(source, np.ndarray):
            return from_numpy(source.shape, source.dtype)
        elif isinstance(source, list):
            # TODO: um yeah, we'd don't actually want to do this
            cast = np.array(source)
            return from_numpy(cast.shape, cast.dtype)
        else:
            return dynamic

    @staticmethod
    def check_datashape(source, given_dshape):
        """
        Does the user specified dshape make sense for the given
        source.
        """
        # TODO
        return True

    def repr_data(self):
        return carray.array2string(self.ca)

    @classmethod
    def empty(self, dshape):
        """ Create a CArraySource from a datashape specification,
        downcasts into Numpy dtype and shape tuples if possible
        otherwise raises an exception.
        """
        shape, dtype = from_numpy(dshape)
        return CArraySource(carray([], dtype))

    def read(self, offset, nbytes):
        """ The future read interface for the CArray.  """
        raise NotImplementedError

    def __repr__(self):
        return 'CArray(ptr=%r)' % id(self.ca)


class ArraySource(ByteProvider):
    """
    The buffer from a Numpy array used a data source

    Only used for bootstrapping. Will be removed later.
    """

    read_capabilities  = CONTIGUOUS
    write_capabilities = CONTIGUOUS

    def __init__(self, lst):
        self.na = np.array(lst)

    @staticmethod
    def infer_datashape(source):
        """
        The user has only provided us with a Python object (
        could be a buffer interface, a string, a list, list of
        lists, etc) try our best to infer what the datashape
        should be in the context of this datasource.
        """
        if isinstance(source, np.ndarray):
            return from_numpy(source.shape, source.dtype)
        elif isinstance(source, list):
            # TODO: um yeah, we'd don't actually want to do this
            cast = np.array(source)
            return from_numpy(cast.shape, cast.dtype)
        else:
            return dynamic

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


class PythonSource(ByteProvider):
    """
    Work with a immutable Python list as if it were byte interfaces.
    """
    read_capabilities  = CONTIGUOUS | CHUNKED | STREAM
    write_capabilities = CONTIGUOUS | CHUNKED | STREAM

    def __init__(self, pyobject, type=list):
        self.pytype = type
        self.pyboject = pyobject

    @staticmethod
    def infer_datashape(source):
        # TODO: more robust
        return pyobj

    def read(self, offset, nbytes):
        # Array types
        # ===========
        if self.pytype == list:
            return self.lst[offset:nbytes]

        # Simple types
        # ============
        elif self.pytype == int:
            return self.pyobject

        elif self.pytype == float:
            return self.pyobject

        elif self.pytype == str:
            return self.pyobject

        else:
            raise TypeError("Don't know how to cast PythonSource")

    def __repr__(self):
        return 'PyObject(ptr=%r)' % id(self.lst)


class ByteSource(ByteProvider):
    """
    A raw block of memory layed out in raditional NumPy fashion.
    """

    read_capabilities  = CONTIGUOUS | CHUNKED | STREAM
    write_capabilities = CONTIGUOUS | CHUNKED | STREAM

    def __init__(self, bits):
        # TODO: lazy
        self.bits = bytearray(bits)
        self.mv   = memoryview(self.bits)

    def read(self, offset, nbytes):
        return self.mv[offset: offset+nbytes]

    @staticmethod
    def infer_datashape(source):
        """
        Just passing in bytes probably means string unless
        dshape otherwise specified.
        """
        return string

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

    read_capabilities  = CHUNKED | STREAM
    write_capabilities = CHUNKED | STREAM

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
