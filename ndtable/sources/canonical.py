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


Abstract               Image               Numpy
========               --------            --------

 Space                 Image               Array
   |                    |                   |
 Subspace              Pixel               Axis
   |                    |                   |
 Byte Descriptor       File Descriptor     Memoryview
   |                    |                   |
 Byte Provider         PNG Source          Memory Source
   |                    |                   |
 Physical Substrate    Disk                Memory

"""
import socket
from ndtable.byteproto import CONTIGIOUS, STRIDED, STREAM, READ, WRITE
from ndtable.bytei import ByteProvider
from ndtable.datashape import Fixed, pyobj
from ndtable.datashape.coretypes import CType

# TODO: rework hierarchy
class Source(ByteProvider):

    def calculate(self, ntype):
        """
        Provide information about the structure of the underlying data
        that is knowable a priori.

        May or may not be dependent on the passed ntype value depending
        on the source.
        """
        raise NotImplementedError()

class PythonSource(Source):
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
        return pyobj*Fixed(len(self.lst))

    def read(self, offset, nbytes):
        return self.lst[offset:nbytes]

    def write(self, offset, nbytes):
        return

    def __repr__(self):
        return 'PyObject(ptr=%r)' % id(self.lst)

class ByteSource(Source):
    """
    Allocate a memory block, explicitly outside of the Python
    heap.
    """

    read_capabilities  = CONTIGIOUS | STRIDED | STREAM
    write_capabilities = CONTIGIOUS | STRIDED | STREAM

    def __init__(self, bits):
        self.bits = bytearray(bits)

    def read(self, offset, nbytes):
        return memoryview(self.bits)[offset: offset+nbytes]

    def calculate(self, ntype):
        if type(ntype) is CType:
            size = ntype.size()
            assert len(self.bits) % size == 0, \
                "Not a multiple of type: %s" % ntype
            return ntype * Fixed(len(self.bits) / size)
        else:
            raise ValueError("Cannot interpret raw bytes as\
            anything but native type.")

    def write(self, offset, wbytes):
        pass

    def __alloc__(self, itemsize, size):
        pass

    def __dealloc__(self):
        pass

class FileSource(Source):
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

class SocketSource(Source):
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

class ImageSource(Source):

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
