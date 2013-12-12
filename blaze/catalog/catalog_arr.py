from __future__ import absolute_import
import os
from os import path
import yaml, csv
from dynd import nd, ndt
import blaze

def load_blaze_array(conf, dir):
    """Loads a blaze array from the catalog configuration and catalog path"""
    # This is a temporary hack, need to transition to using the
    # deferred data descriptors for various formats.
    fsdir = conf.get_fsdir(dir)
    if not path.isfile(fsdir + '.array'):
        raise RuntimeError('Could not find blaze array description file %r'
                % (fsdir + '.array'))
    with open(fsdir + '.array') as f:
        arrmeta = yaml.load(f)
    tp = arrmeta['type']
    imp = arrmeta['import']
    ds = arrmeta['datashape']

    if tp == 'csv':
        with open(fsdir + '.csv', 'rb') as f:
            rd = csv.reader(f)
            if imp.get('headers', False):
                # Skip the header line
                next(rd)
            dat = list(rd)
        arr = nd.array(dat, ndt.type(ds))[:]
        return blaze.array(arr)
    elif tp == 'json':
        arr = nd.parse_json(ds, nd.memmap(fsdir + '.json'))
        return blaze.array(arr)
    else:
        raise ValueError('Unsupported array type %r from blaze catalog entry %r' %
                (tp, dir))