import os, os.path
import uuid

from urlparse import urlparse
from params import params
from sources.chunked import CArraySource
from table import NDArray, Array

def open(uri=None):

    if uri is None:
        source = CArraySource()
    else:
        uri = urlparse(uri)

        if uri.scheme == 'carray':
            path = os.path.join(uri.netloc, uri.path[1:])
            parms = params(storage=path)
            source = CArraySource(params=parms)
        #elif uri.scheme == 'tcp':
            #byte_interface = SocketSource()
        else:
            # Default is to treak the URI as a regular path
            parms = params(storage=uri.path)
            source = CArraySource(params=parms)

    # Don't want a deferred array (yet)
    # return NDArray(source)
    return Array(source)

