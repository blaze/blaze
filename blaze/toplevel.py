import os, os.path
import uuid

from urlparse import urlparse
from params import params, to_cparams
from sources.chunked import CArraySource
from table import NDArray, Array
from blaze.datashape.coretypes import from_numpy, to_numpy
from blaze import carray

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
            # Default is to treat the URI as a regular path
            parms = params(storage=uri.path)
            source = CArraySource(params=parms)

    # Don't want a deferred array (yet)
    # return NDArray(source)
    return Array(source)

def zeros(dshape, params=None):
    """ Create an Array and fill it with zeros.
    """
    shape, dtype = to_numpy(dshape)
    cparams, rootdir, format_flavor = to_cparams(params)
    if rootdir is not None:
        carray.zeros(shape, dtype, rootdir=rootdir, cparams=cparams)
        return open(rootdir)
    else:
        source = CArraySource(carray.zeros(shape, dtype, cparams=cparams),
                              params=params)
        return Array(source)

def ones(dshape, params=None):
    """ Create an Array and fill it with ones.
    """
    shape, dtype = to_numpy(dshape)
    cparams, rootdir, format_flavor = to_cparams(params)
    if rootdir is not None:
        carray.ones(shape, dtype, rootdir=rootdir, cparams=cparams)
        return open(rootdir)
    else:
        source = CArraySource(carray.ones(shape, dtype, cparams=cparams),
                              params=params)
        return Array(source)

