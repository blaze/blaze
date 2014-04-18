from __future__ import absolute_import, division, print_function

from functools import partial
from .csv import *
from .json import *
from .hdf5 import *
from .filesystem import *
from .sql import *
from glob import glob
import gzip
from ..compatibility import urlopen

__all__ = ['resource']

filetypes = {'csv': CSV,
             'tsv': CSV,
             'json': JSON,
             'h5': HDF5,
             'hdf5': HDF5}

opens = {'http': urlopen,
         'https': urlopen,
        #'ssh': paramiko.open?
         }

def resource(uri, **kwargs):
    """ Get data resource from universal resource indicator

    """
    descriptor = None
    args = []

    if '::' in uri:
        uri, datapath = uri.rsplit('::')
        args.insert(0, datapath)

    extensions = uri.split('.')
    if extensions[-1] == 'gz':
        kwargs['open'] = kwargs.get('open', gzip.open)
        extensions.pop()
    descriptor = filetypes.get(extensions[-1], None)

    if '://' in uri:
        protocol, _ = uri.split('://')
        if protocol in opens:
            kwargs['open'] = kwargs.get('open', opens[protocol])
        if 'sql' in protocol:
            descriptor = SQL

    try:
        filenames = glob(uri)
    except:
        filenames = []
    if len(filenames) > 1:
        args = [partial(descriptor, *args)]  # pack sub descriptor into args
        descriptor = Files

    if descriptor:
        return descriptor(uri, *args, **kwargs)

    raise ValueError('Unknown resource type\n\t%s' % uri)
