import os
import uuid

from urlparse import urlparse
from params import params
from adaptors.chunked import CArraySource

def open(uri=None):

    if uri is None:
        source = CArraySource()
    else:
        uri = urlparse(uri)

        if uri.scheme == 'carray':
            params(storage=uri.netloc)
            source = CArraySource(params=params)
        #elif uri.scheme == 'tcp':
            #byte_interface = SocketSource()

    #return NDArray
