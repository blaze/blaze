from __future__ import absolute_import, division, print_function

__all__ = ['RArray']

from dynd import nd, ndt
from . import requests
from ...catalog.blaze_url import add_indexers_to_url
from ...objects.constructors import array
from ... import datashape


class RArray(object):
    def __init__(self, url, dshape=None):
        self.url = url
        if dshape is None:
            self.dshape = datashape.dshape(requests.get_remote_datashape(url))
        else:
            self.dshape = datashape.dshape(dshape)

    def __repr__(self):
        return ('Remote Blaze Array\nurl: %s\ndshape: %s\n'
                % (self.url, self.dshape))

    def __getattr__(self, name):
        ds = self.dshape
        if isinstance(self.dshape, datashape.DataShape):
            ds = ds[-1]
        if isinstance(self.dshape, datashape.Record) and name in ds.names:
            return rarray(self.url + '.' + name)
        else:
            raise AttributeError(('Blaze remote array does not ' +
                                  'have attribute "%s"') % name)

    def __getitem__(self, key):
        return rarray(add_indexers_to_url(self.url, (key,)))

    def get_dynd(self):
        """Downloads the data and returns a local in-memory nd.array"""
        j = requests.get_remote_json(self.url)
        tp = ndt.type(str(self.dshape))
        return nd.parse_json(tp, j)

    def get_data(self):
        """Downloads the data and returns a local in-memory blaze.array"""
        return array(self.get_dynd())
