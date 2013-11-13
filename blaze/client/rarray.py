# -*- coding: utf-8 -*-
__all__ = ['rarray']

import requests
from ..catalog.blaze_url import add_indexers_to_url
from dynd import nd, ndt
from ..constructors import array

class rarray:
    def __init__(self, url, dshape=None):
        self.url = url
        if dshape is None:
            self.dshape = requests.get_remote_datashape(url)
        else:
            self.dshape = dshape
        self.dtype = ndt.type(self.dshape)

    def __repr__(self):
        return 'Remote Blaze Array\nurl: %s\ndshape: %s\n' % \
                        (self.url, self.dshape)

    def __getattr__(self, name):
        if name in self.dtype.property_names:
            return rarray(self.url + '.' + name)
        else:
            raise AttributeError('Blaze remote array does not have attribute "%s"' % name)

    def __getitem__(self, key):
        return rarray(add_indexers_to_url(self.url, (key,)))

    def get_dynd(self):
        """Downloads the data and returns a local in-memory nd.array"""
        j = requests.get_remote_json(self.url)
        return nd.parse_json(self.dtype, j)

    def get_data(self):
        """Downloads the data and returns a local in-memory blaze.array"""
        return array(self.get_dynd())

