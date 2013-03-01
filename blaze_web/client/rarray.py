# -*- coding: utf-8 -*-
__all__ = ['rarray']

import requests
from blaze_web.common.blaze_url import add_indexers_to_url
from dynd import nd, ndt

class rarray:
    def __init__(self, url):
        self.url = url
        self.dshape = requests.get_remote_datashape(url)
        self.dtype = nd.dtype(self.dshape)

    def __repr__(self):
        return 'Remote Blaze Array\n  url: %s\n  dshape: %s\n' % \
                        (self.url, self.dshape)

    def __getattr__(self, name):
        if name in self.dtype.property_names:
            return rarray(self.url + '.' + name)
        else:
            raise AttributeError('Blaze remote array does not have attribute "%s"' % name)

    def __getitem__(self, key):
        if type(key) in [int, long, slice]:
            key = (key,)
        return rarray(add_indexers_to_url(self.url, key))

    def get_data(self):
        """Downloads the data and returns a local in-memory ndobject"""
        j = requests.get_remote_json(self.url)
        return nd.parse_json(self.dshape, j)
