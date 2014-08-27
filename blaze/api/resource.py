from __future__ import absolute_import, division, print_function
from dynd import nd
import pandas as pd
import os
from glob import glob
from ..dispatch import dispatch
from ..data import DataDescriptor, CSV
from blaze.compute.chunks import ChunkIndexable

class ChunkList(ChunkIndexable):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)


def resource(uri, **kwargs):
    if '*' in uri:
        uris = sorted(glob(uri))
        return ChunkList(list(map(resource, uris)))

    csv_extensions = ['csv', 'data', 'txt', 'dat']
    if os.path.isfile(uri) and uri.split('.')[-1] in csv_extensions:
        return CSV(uri, **kwargs)
    if (os.path.isfile(uri) and uri.endswith("gz")
                            and uri.split('.')[-2] in csv_extensions):
        return CSV(uri, open=gzip.open, **kwargs)
    else:
        raise NotImplementedError(
            "Blaze can't read '%s' yet!" % uri)
