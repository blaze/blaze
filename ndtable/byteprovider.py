"""
ByteProvider base class.
"""

from byteproto import READ, WRITE
from datadescriptor import DataDescriptor

class ByteProvider(object):
    """

    The ByteProvider provides the high-level API operations on those bytes
    that abstracts away the notion of whether the object is contiguous,
    chunked, or streamed and can extract or write bytes in any case.

    If the DataDescriptor supports an operation natively then it can perform
    it in a single "instruction", if it does not then the byte interface
    will devise a way to do the operation as a sequence of instructions.
    """

    def getbuffer(self):
        raise NotImplementedError

    def read_desc(self):
        """ Returns the naive descriptor, which will be
        speciazlied by execution """
        raise NotImplementedError

    def write_desc(self):
        """ Returns the naive descriptor, which will be
        speciazlied by execution """
        raise NotImplementedError

    def has_op(self, op, method):
        if op == READ:
            return method & self.read_capabilities
        if op == WRITE:
            return method & self.write_capabilities

    @classmethod
    def empty(self, datashape):
        """
        Create a empty container conforming to the given
        datashape. Requires the ACCESS_ALLOC flag.
        """
        raise NotImplementedError
