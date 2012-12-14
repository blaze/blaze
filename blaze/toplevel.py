import os, os.path

from urlparse import urlparse
from params import params, to_cparams
from params import params as _params
from sources.sql import SqliteSource
from sources.chunked import CArraySource

from table import NDArray, Array
from blaze.datashape.coretypes import from_numpy, to_numpy, TypeVar, Fixed
from blaze import carray, dshape as _dshape

import numpy as np

def open(uri=None):
    """Open a Blaze object via an `uri` (Uniform Resource Identifier).
    
    Parameters
    ----------
    uri : str
        Specifies the URI for the Blaze object.  It can be a regular file too.

    Returns
    -------
    out : an Array object.

    """

    if uri is None:
        source = CArraySource()
    else:
        uri = urlparse(uri)

        if uri.scheme == 'carray':
            path = os.path.join(uri.netloc, uri.path[1:])
            parms = params(storage=path)
            source = CArraySource(params=parms)

        elif uri.scheme == 'sqlite':
            path = os.path.join(uri.netloc, uri.path[1:])
            parms = params(storage=path or None)
            source = SqliteSource(params=parms)

        else:
            # Default is to treat the URI as a regular path
            parms = params(storage=uri.path)
            source = CArraySource(params=parms)

    # Don't want a deferred array (yet)
    # return NDArray(source)
    return Array(source)

# These are like NumPy equivalent except that they can allocate
# larger than memory.

def zeros(dshape, params=None):
    """ Create an Array and fill it with zeros.

    Parameters
    ----------
    dshape : str, blaze.dshape instance
        Specifies the datashape of the outcome object.
    params : blaze.params object
        Any parameter supported by the backend library.

    Returns
    -------
    out : an Array object.

    """
    if isinstance(dshape, basestring):
        dshape = _dshape(dshape)
    shape, dtype = to_numpy(dshape)
    cparams, rootdir, format_flavor = to_cparams(params or _params())
    if rootdir is not None:
        carray.zeros(shape, dtype, rootdir=rootdir, cparams=cparams)
        return open(rootdir)
    else:
        source = CArraySource(carray.zeros(shape, dtype, cparams=cparams),
                              params=params)
        return Array(source)

def ones(dshape, params=None):
    """ Create an Array and fill it with ones.

    Parameters
    ----------
    dshape : str, blaze.dshape instance
        Specifies the datashape of the outcome object.
    params : blaze.params object
        Any parameter supported by the backend library.

    Returns
    -------
    out : an Array object.

    """
    if isinstance(dshape, basestring):
        dshape = _dshape(dshape)
    shape, dtype = to_numpy(dshape)
    cparams, rootdir, format_flavor = to_cparams(params or _params())
    if rootdir is not None:
        carray.ones(shape, dtype, rootdir=rootdir, cparams=cparams)
        return open(rootdir)
    else:
        source = CArraySource(carray.ones(shape, dtype, cparams=cparams),
                              params=params)
        return Array(source)

def fromiter(iterable, dshape, params=None):
    """ Create an Array and fill it with values from `iterable`.

    Parameters
    ----------
    iterable : iterable object
        An iterable object providing data for the carray.
    dshape : str, blaze.dshape instance
        Specifies the datashape of the outcome object.  Only 1d shapes
        are supported right now. When the `iterator` should return an
        unknown number of items, a ``TypeVar`` can be used.
    params : blaze.params object
        Any parameter supported by the backend library.

    Returns
    -------
    out : an Array object.

    """
    if isinstance(dshape, basestring):
        dshape = _dshape(dshape)
    shape, dtype = dshape.parameters[:-1], dshape.parameters[-1]
    # Check the shape part
    if len(shape) > 1:
        raise ValueError("shape can be only 1-dimensional")
    length = shape[0]
    count = -1
    if type(length) == TypeVar:
        count = -1
    elif type(length) == Fixed:
        count = length.val

    dtype = dtype.to_dtype()
    # Now, create the Array itself (using the carray backend)
    cparams, rootdir, format_flavor = to_cparams(params or _params())
    if rootdir is not None:
        carray.fromiter(iterable, dtype, count=count,
                        rootdir=rootdir, cparams=cparams)
        return open(rootdir)
    else:
        ica = carray.fromiter(iterable, dtype, count=count, cparams=cparams)
        source = CArraySource(ica, params=params)
        return Array(source)

def loadtxt(filetxt, storage):
    """ Convert txt file into Blaze native format """
    Array(np.loadtxt(filetxt), params=params(storage=storage))
