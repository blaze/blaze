from __future__ import absolute_import

# This file contains the part of the blaze API dealing with URIs. The
# "URI API". In Blaze persistence is provided by the means of this URI
# API, that allows specifying a "location" for an array as an URI. 
#
# The URI API allows:
#
# - saving existing arrays to an URI.
#
# - loading an array into memory from an URI.
#
# - opening an URI as an array.
#
# - dropping the contents of a given URI.

from ._api_helpers import _normalize_dshape
from .datashape import to_numpy, to_dtype
from .py3help import urlparse
from . import blz
from .datadescriptor import (BLZDataDescriptor)
from .array import Array

# ----------------------------------------------------------------------
# Some helper functions to workaround quirks
 
# XXX A big hack for some quirks in current datashape. The next deals
# with the cases where the shape is not present like in 'float32'
def _to_numpy(ds):
    res = to_numpy(ds)
    res = res if type(res) is tuple else ((), to_dtype(ds))
    return res


# ----------------------------------------------------------------------
# The actual URI API

def save(array, uri):
    """save an array to an URI"""
    pass


def load(uri):
    """load and array into memory from an URI"""
    pass


def open(uri):
    """Open an existing persistent array.

    Parameters
    ----------
    uri : URI string
        The URI of where the array is stored (e.g. blz://myfile.blz).

    Returns
    -------
    out: a concrete blaze array.

    Notes
    -----
    Only the BLZ format is supported currently.

    """
    uri = urlparse.urlparse(uri)
    path = uri.netloc + uri.path
    d = blz.open(rootdir=path)
    dd = BLZDataDescriptor(d)
    return Array(dd)


def drop(uri):
    """removing an URI"""
    pass


# Persistent constructors:
def create(uri, dshape, caps={'efficient-append': True}):
    """Create a 0-length persistent array.

    Parameters
    ----------
    uri : URI string
        The URI of where the array will be stored (e.g. blz://myfile.blz).

    dshape : datashape
        The datashape for the resulting array.

    caps : capabilities dictionary
        A dictionary containing the desired capabilities of the array.

    Returns
    -------
    out: a concrete blaze array.

    Notes
    -----

    The shape part of the `dshape` is ignored.  This should be fixed
    by testing that the shape is actually empty.

    Only the BLZ format is supported currently.

    """
    dshape = _normalize_dshape(dshape)

    # Only BLZ supports efficient appends right now
    shape, dt = _to_numpy(dshape)
    shape = (0,) + shape  # the leading dimension will be 0
    uri = urlparse.urlparse(uri)
    path = uri.netloc + uri.path
    if 'efficient-append' in caps:
        dd = BLZDataDescriptor(blz.zeros(shape, dtype=dt, rootdir=path))
    elif 'efficient-write' in caps:
        raise ValueError('efficient-write objects not supported for '
                         'persistence')
    else:
        # BLZ will be the default
        dd = BLZDataDescriptor(blz.zeros(shape, dtype=dt, rootdir=path))
    return Array(dd)


def create_fromiter(uri, dshape, iterator):
    """create persistent array at the URI initialized with the
    iterator iterator"""
    pass
