from __future__ import absolute_import
import operator
import contextlib

from .data_descriptor import IDataDescriptor
from ... import datashape
from ...datashape import dshape
from ...error import ArrayWriteError

from dynd import nd, ndt, _lowlevel


def dynd_descriptor_iter(dyndarr):
    for el in dyndarr:
        yield DyNDDataDescriptor(el)

class DyNDDataDescriptor(IDataDescriptor):
    """
    A Blaze data descriptor which exposes a DyND array.
    """
    def __init__(self, dyndarr):
        if not isinstance(dyndarr, nd.array):
            raise TypeError('object is not a dynd array, has type %s' %
                            type(dyndarr))
        self._dyndarr = dyndarr
        self._dshape = dshape(nd.dshape_of(dyndarr))

    @property
    def is_concrete(self):
        """Returns True, dynd arrays are concrete.
           TODO: Maybe not always, if the dynd array has an expression type?
        """
        return True

    def dynd_arr(self):
        return self._dyndarr

    @property
    def dshape(self):
        return self._dshape

    @property
    def writable(self):
        return self._dyndarr.access_flags == 'readwrite'

    @property
    def immutable(self):
        return self._dyndarr.access_flags == 'immutable'

    def __array__(self):
        return nd.as_numpy(self.dynd_arr())

    def __len__(self):
        return len(self._dyndarr)

    def __getitem__(self, key):
        return DyNDDataDescriptor(self._dyndarr[key])

    def __setitem__(self, key, value):
        # TODO: This is a horrible hack, we need to specify item setting
        #       via well-defined interfaces, not punt to another system.
        self._dyndarr.__setitem__(key, value)

    def __iter__(self):
        return dynd_descriptor_iter(self._dyndarr)
