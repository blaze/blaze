from __future__ import absolute_import, division, print_function

from .csv import *
from .json import *
from .hdf5 import *
from .filesystem import *
from .sql import *
from glob import glob
import gzip
import urllib2

__all__ = ['resource']

filetypes = {'csv': CSV,
             'tsv': CSV,
             'json': JSON,
             'h5': HDF5,
             'hdf5': HDF5}

opens = {'http': urllib2.urlopen,
         'https': urllib2.urlopen,
        #'ssh': paramiko.open?
         }

def resource(uri, **kwargs):
    """ Get data resource from universal resource indicator

    """
    extensions = uri.split('.')
    if extensions[-1] == 'gz':
        open = gzip.open
        kwargs['open'] = kwargs.get('open', gzip.open)
        extension = extensions[-2]
    else:
        extension = extensions[-1]

    if extension not in filetypes:
        raise ValueError('Unknown resource type\n\t%s' % extension)

    descriptor = filetypes[extension]

    return descriptor(uri, **kwargs)
