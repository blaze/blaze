"""URI API

This file contains the part of the blaze API dealing with URIs. The
"URI API". In Blaze persistence is provided by the means of this URI
API, that allows specifying a "location" for an array as an URI.

The URI API allows:

- saving existing arrays to an URI.

- loading an array into memory from an URI.

- opening an URI as an array.

- dropping the contents of a given URI.

"""

from __future__ import absolute_import, division, print_function

import os
import warnings

from datashape import to_numpy, to_numpy_dtype
import blz

from ..py2help import urlparse
from ..datadescriptor import (BLZDataDescriptor, CSVDataDescriptor,
                              JSONDataDescriptor, HDF5DataDescriptor)
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

    Class to host parameters for persistence properties.

    Parameters
    ----------
    uri : string
        The URI where the data set will be stored.
    mode : string ('r'ead, 'a'ppend)
        The mode for creating/opening the storage.
    permanent : bool
        Whether this file should be permanent or not.

    Examples
    --------
    >>> store = Storage('blz-store.blz')

    """

    SUPPORTED_FORMATS = ('json', 'csv', 'blz', 'hdf5')

    @property
    def uri(self):
        """The URI for the data set."""
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
        """Returns a blz path for a given uri."""
        return self._path

    def __init__(self, uri, mode='a', permanent=True, format=None):
        if not isinstance(uri, str):
            raise ValueError("`uri` must be a string.")
        self._uri = uri
        self._format = self._path = ""
        self._set_format_and_path_from_uri(uri, format)
        self._mode = mode
        if not permanent:
            raise ValueError(
                "`permanent` set to False is not supported yet.")
        self._permanent = permanent

    def __repr__(self):
        args = ["uri=%s" % self._uri, "mode=%s" % self._mode]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(args))

    def _set_format_and_path_from_uri(self, uri, format=None):
        """Parse the uri into the format and path"""
        up = urlparse.urlparse(self._uri)
        if up.scheme in self.SUPPORTED_FORMATS:
            warnings.warn("Blaze no longer uses file type in network protocol field of the uri. "
                          "Please use format kwarg.", DeprecationWarning)
        self._path = up.netloc + up.path
        if os.name == 'nt' and len(up.scheme) == 1:
            # This is a workaround for raw windows paths like
            # 'C:/x/y/z.csv', for which urlparse parses 'C' as
            # the scheme and '/x/y/z.csv' as the path.
            self._path = uri
        if not self._path:
            raise ValueError("Unable to extract path from uri: %s", uri)
        _, extension = os.path.splitext(self._path)
        extension = extension.strip('.')

        # Support for deprecated format in url network scheme
        format_from_up = None
        if up.scheme in self.SUPPORTED_FORMATS:
            format_from_up = up.scheme
        if format and format_from_up != format_from_up:
            raise ValueError("URI scheme and file format do not match. Given uri: %s, format: %s" %
                             (up.geturl(), format))

        # find actual format
        if format:
            self._format = format
        elif format_from_up:
            self._format = format_from_up
        elif extension:
            self._format = extension
        else:
            raise ValueError("Cannot determine format from: %s" % uri)

        if self._format not in self.SUPPORTED_FORMATS:
            raise ValueError("`format` '%s' is not supported." % self._format)


def _persist_convert(persist):
    if not isinstance(persist, Storage):
        if isinstance(persist, str):
            persist = Storage(persist)
        else:
            raise ValueError('persist argument must be either a'
                             'URI string or Storage object')
    return persist


# ----------------------------------------------------------------------
# The actual API specific for persistence.
# Only BLZ, HDF5, CSV and JSON formats are supported currently.

def from_blz(persist, **kwargs):
    """Open an existing persistent BLZ array.

    Parameters
    ----------
    persist : a Storage instance
        The Storage instance specifies, among other things, path of
        where the array is stored.
    kwargs : a dictionary
        Put here different parameters depending on the format.

    Returns
    -------
    out: a concrete blaze array.

    """
    persist = _persist_convert(persist)
    d = blz.barray(rootdir=persist.path, **kwargs)
    dd = BLZDataDescriptor(d)
    return Array(dd)

def from_csv(persist, **kwargs):
    """Open an existing persistent CSV array.

    Parameters
    ----------
    persist : a Storage instance
        The Storage instance specifies, among other things, path of
        where the array is stored.
    kwargs : a dictionary
        Put here different parameters depending on the format.

    Returns
    -------
    out: a concrete blaze array.

    """
    persist = _persist_convert(persist)
    dd = CSVDataDescriptor(persist.path, **kwargs)
    return Array(dd)

def from_json(persist, **kwargs):
    """Open an existing persistent JSON array.

    Parameters
    ----------
    persist : a Storage instance
        The Storage instance specifies, among other things, path of
        where the array is stored.
    kwargs : a dictionary
        Put here different parameters depending on the format.

    Returns
    -------
    out: a concrete blaze array.

    """
    persist = _persist_convert(persist)
    dd = JSONDataDescriptor(persist.path, **kwargs)
    return Array(dd)

def from_hdf5(persist, **kwargs):
    """Open an existing persistent HDF5 array.

    Parameters
    ----------
    persist : a Storage instance
        The Storage instance specifies, among other things, path of
        where the array is stored.
    kwargs : a dictionary
        Put here different parameters depending on the format.

    Returns
    -------
    out: a concrete blaze array.

    """
    persist = _persist_convert(persist)
    dd = HDF5DataDescriptor(persist.path, **kwargs)
    return Array(dd)

def drop(persist):
    """Remove a persistent storage."""

    persist = _persist_convert(persist)

    if persist.format == 'blz':
        from shutil import rmtree
        rmtree(persist.path)
    elif persist.format in ('csv', 'json', 'hdf5'):
        import os
        os.unlink(persist.path)
