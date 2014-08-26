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

    if os.path.isfile(uri) and uri.endswith("csv"):
        return CSV(uri, **kwargs)
    else:
        raise NotImplementedError(
            "Blaze can't read '%s' yet!" % uri)
