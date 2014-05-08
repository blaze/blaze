from __future__ import absolute_import, division, print_function

from functools import partial
from .csv import *
from .json import *
from .hdf5 import *
from .filesystem import *
from .sql import *
from glob import glob
import gzip
from ..compatibility import urlopen, _strtypes

__all__ = ['resource', 'copy']

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

    Supports the following logic:

    *   Infer data format based on the file extension (.csv, .json. .hdf5)
    *   Use ``gzip.open`` if files end in ``.gz`` extension (csv, json only)
    *   Use ``urlopen`` if web protocols detected (http, https)
    *   Use SQL if text ``sql`` found in protocol string

    URI may be in any of the following forms

    >>> uri = '/path/to/data.csv'                     # csv, json, etc...
    >>> uri = '/path/to/data.json.gz'                 # handles gzip
    >>> uri = '/path/to/*/many*/data.*.json'          # glob string - many files
    >>> uri = '/path/to/data.hdf5::/path/within/hdf5' # HDF5 path :: datapath
    >>> uri = 'postgresql://sqlalchemy.uri::tablename'# SQLAlchemy :: tablename
    >>> uri = 'http://api.domain.com/data.json'       # Web requests

    Note that this follows standard ``protocol://path`` syntax.  In cases where
    more information is needed, such as an HDF5 datapath or a SQL table name
    the additional information follows two colons `::` as in the following

        /path/to/data.hdf5::/datapath
    """
    descriptor = None
    args = []
    in_uri = uri

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
        resources = [resource(in_uri.replace(uri, filename), **kwargs)
                        for filename in filenames]
        return Stack(resources)

    if descriptor:
        return descriptor(uri, *args, **kwargs)

    raise ValueError('Unknown resource type\n\t%s' % uri)


def copy(src, dest, **kwargs):
    """ Copy content from one data descriptor to another """
    dest.extend_chunks(src.chunks(**kwargs))
