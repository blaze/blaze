import os
import uuid
from urlparse import urlparse
from adaptors.canonical import MemoryAdaptor, FileAdaptor, SocketAdaptor

class Blaze(object):
    """
    The blaze scheduler.
    """

    def __init__(self):
        self.pid = os.getpid()
        self.managed = {}

    def open(self, uri=None):
        byte_interface = None

        if uri is None:
            byte_interface = MemoryAdaptor()
        else:
            uri = urlparse(uri)

            if uri.scheme == 'file':
                byte_interface = FileAdaptor(uri.netloc)
            elif uri.scheme == 'tcp':
                byte_interface = SocketAdaptor()

        if byte_interface:
            self.managed[str(uuid.uuid4())] = byte_interface
            return byte_interface
        else:
            raise NotImplementedError()
