from __future__ import absolute_import, division, print_function

from dynd import nd

from .core import DataDescriptor


class DyND(DataDescriptor):
    deferred = False
    persistent = False
    appendable = False
    remote = False

    def __init__(self, arr):
        self.arr = arr

    @property
    def immutable(self):
        return self.arr.access_flags == 'immutable'

    @property
    def _dshape(self):
        return nd.dshape_of(self.arr)

    def _iter(self):
        return iter(self.arr)

    def _getitem(self, key):
        return self.storage[key]

    def _chunks(self, blen=100):
        for i in range(0, len(self.arr), blen):
            start = i
            stop = min(i + blen, len(self.arr))
            yield self.arr[start:stop]

    def dynd_arr(self):
        return self.arr
