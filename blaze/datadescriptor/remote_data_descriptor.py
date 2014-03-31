from __future__ import absolute_import, division, print_function

import datashape
from ..catalog.blaze_url import add_indexers_to_url
from .data_descriptor import I_DDesc, Capabilities
from dynd import nd, ndt


class Remote_DDesc(I_DDesc):
    """
    A Blaze data descriptor which exposes an array on another
    server.
    """

    def __init__(self, url, dshape=None):
        from ..io.client import requests
        self.url = url
        if dshape is None:
            self._dshape = datashape.dshape(requests.get_remote_datashape(url))
        else:
            self._dshape = datashape.dshape(dshape)

    @property
    def dshape(self):
        return self._dshape

    @property
    def capabilities(self):
        """The capabilities for the remote data descriptor."""
        return Capabilities(
            # treat remote arrays as immutable (maybe not?)
            immutable = True,
            # TODO: not sure what to say here
            deferred = False,
            # persistent on the remote server
            persistent = True,
            appendable = False,
            remote = True,
            )

    def __repr__(self):
        return 'Remote_DDesc(%r, dshape=%r)' % (self.url, self.dshape)

    def dynd_arr(self):
        from ..io.client import requests
        """Downloads the data and returns a local in-memory nd.array"""
        # TODO: Need binary serialization
        j = requests.get_remote_json(self.url)
        tp = ndt.type(str(self.dshape))
        return nd.parse_json(tp, j)

    def __len__(self):
        ds = self.dshape
        if isinstance(ds, datashape.DataShape):
            ds = ds[-1]
            if isinstance(ds, datashape.Fixed):
                return int(ds)
        raise AttributeError('the datashape (%s) of this data descriptor has no length' % ds)

    def __getitem__(self, key):
        return Remote_DDesc(add_indexers_to_url(self.url, (key,)))

    def getattr(self, name):
        ds = self.dshape
        if isinstance(ds, datashape.DataShape):
            ds = ds[-1]
        if isinstance(ds, datashape.Record) and name in ds.names:
            return Remote_DDesc(self.url + '.' + name)
        else:
            raise AttributeError(('Blaze remote array does not ' +
                                  'have attribute "%s"') % name)

    def __iter__(self):
        raise NotImplementedError('remote data descriptor iterator unimplemented')
