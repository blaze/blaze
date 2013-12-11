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

from ..datashape import to_numpy, to_numpy_dtype
from ..py2help import urlparse
from . import blz
from ..datadescriptor import (BLZDataDescriptor, CSVDataDescriptor,
                              JSONDataDescriptor)
from ..objects.array import Array

# ----------------------------------------------------------------------
# Some helper functions to workaround quirks

# XXX A big hack for some quirks in current datashape. The next deals
# with the cases where the shape is not present like in 'float32'
def _to_numpy(ds):
    res = to_numpy(ds)
    res = res if type(res) is tuple else ((), to_numpy_dtype(ds))
    return res


class Storage(object):
    """
    Storage(uri, mode='a', permanent=True)

    Class to host parameters for persistency properties.

    Parameters
    ----------
    uri : string
        The URI where the dataset will be stored.
    mode : string ('r'ead, 'a'ppend)
        The mode for creating/opening the storage.
    permanent : bool
        Whether this file should be permanent or not.

    Examples
    --------
    >>> store = Storage('blz://blz-store')

    """

    @property
    def uri(self):
        """The URI for the dataset."""
        return self._uri

    @property
    def mode(self):
        """The mode for opening the storage."""
        return self._mode

    @property
    def format(self):
        """The format used for storage."""
        return self._format

    @property
    def permanent(self):
        """Whether this file should be permanent or not."""
        return self._permanent

    @property
    def path(self):
        """ returns a blz path for a given uri """
        return self._path

    def __init__(self, uri, mode='r', permanent=True):
        if not isinstance(uri, str):
            raise ValueError("`uri` must be a string.")
        self._uri = uri
        # Parse the uri into the format (URI scheme) and path
        up = urlparse.urlparse(self._uri)
        self._format = format = up.scheme
        if self._format not in ('json', 'csv', 'blz'):
            raise ValueError("`format` '%s' is not supported." % self._format)
        self._path = up.netloc + up.path
        self._mode = mode
        if not permanent:
            raise ValueError(
                "`permanent` set to False is not supported yet.")
        self._permanent = permanent

    def __repr__(self):
        args = ["uri=%s"%self._uri, "mode=%s"%self._mode]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(args))


def _persist_convert(persist):
    if not isinstance(persist, Storage):
        if isinstance(persist, str):
            persist = Storage(persist)
        else:
            raise ValueError('persist argument must be either a'
                             'URI string or Storage object')
    return persist


# ----------------------------------------------------------------------
# The actual API specific for persistence

def open(persist, **kwargs):
    """Open an existing persistent array.

    Parameters
    ----------
    persist : a Storage instance
        The Storage instance specifies, among other things, URI of
        where the array is stored.
    kwargs : a dictionary
        Put here different paramaters depending on the format.

    Returns
    -------
    out: a concrete blaze array.

    Notes
    -----
    Only BLZ, CSV and JSON formats are supported currently.

    """
    persist = _persist_convert(persist)
    if persist.format == 'blz':
        d = blz.barray(rootdir=persist.path, **kwargs)
        dd = BLZDataDescriptor(d)
    elif persist.format == 'csv':
        d = file(persist.path, mode=persist.mode)
        dd = CSVDataDescriptor(d, **kwargs)
    elif persist.format == 'json':
        d = file(persist.path, mode=persist.mode)
        dd = JSONDataDescriptor(d, **kwargs)
    return Array(dd)


def drop(persist):
    """Remove a persistent storage."""

    persist = _persist_convert(persist)

    if persist.format== 'blz':
        try:
            blz.open(rootdir=persist.path)
            from shutil import rmtree
            rmtree(persist.path)
        except RuntimeError:
            # Maybe BLZ should throw other exceptions for this!
            raise Exception("No dataset at uri '%s'" % uri)
    elif persist.format== 'csv':
        import os
        os.unlink(persist.path)
