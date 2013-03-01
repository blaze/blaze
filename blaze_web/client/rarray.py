# -*- coding: utf-8 -*-
__all__ = ['rarray']

import requests
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
        if self.dtype.udtype.kind == 'struct':
            if name in self.dtype.udtype.field_names:
                return rarray(self.url + '.' + name)
        raise AttributeError('Blaze remote array does not have attribute "%s"' % name)