from __future__ import absolute_import, division, print_function

from datashape import dshape
from dynd import nd

from .data_descriptor import DDesc


def dynd_descriptor_iter(dyndarr):
    for el in dyndarr:
        yield DyND_DDesc(el)


class DyND_DDesc(DDesc):
    """
    A Blaze data descriptor which exposes a DyND array.
    """
    def __init__(self, dyndarr):
        if not isinstance(dyndarr, nd.array):
            raise TypeError('object is not a dynd array, has type %s' %
                            type(dyndarr))
        self._dyndarr = dyndarr
        self._dshape = dshape(nd.dshape_of(dyndarr))

    def dynd_arr(self):
        return self._dyndarr

    @property
    def dshape(self):
        return self._dshape

    @property
    def capabilities(self):
        """The capabilities for the dynd data descriptor."""
        return {'immutable': self._dyndarr.access_flags == 'immutable',
                'deferred': False,
                'persistent': False,
                'appendable': False,
                'remote': False}

    def __array__(self):
        return nd.as_numpy(self.dynd_arr())

    def __len__(self):
        return len(self._dyndarr)

    def __getitem__(self, key):
        return DyND_DDesc(self._dyndarr[key])

    def __setitem__(self, key, value):
        # TODO: This is a horrible hack, we need to specify item setting
        #       via well-defined interfaces, not punt to another system.
        self._dyndarr.__setitem__(key, value)

    def __iter__(self):
        return dynd_descriptor_iter(self._dyndarr)

    def getattr(self, name):
        return DyND_DDesc(getattr(self._dyndarr, name))
