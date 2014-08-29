from __future__ import absolute_import, division, print_function
from dynd import nd
import pandas as pd
import os
from glob import glob
import gzip
from ..dispatch import dispatch
from ..data import DataDescriptor, CSV
from blaze.compute.chunks import ChunkIndexable
from ..regex import RegexDispatcher

class ChunkList(ChunkIndexable):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)


resource = RegexDispatcher('resource')

@resource.register('.*\*.*', priority=14)
def resource_glob(uri, **kwargs):
    uris = sorted(glob(uri))
    return ChunkList(list(map(resource, uris)))


@resource.register('.*\.(csv|data|txt|dat)')
def resource_csv(uri, **kwargs):
    return CSV(uri, **kwargs)


@resource.register('.*\.(csv|data|txt|dat)\.gz')
def resource_csv(uri, **kwargs):
    return CSV(uri, open=gzip.open, **kwargs)


@resource.register('.*', priority=1)
def resource_all(uri, *args, **kwargs):
    raise NotImplementedError("Unable to parse uri to data resource: " + uri)
