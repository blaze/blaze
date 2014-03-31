from __future__ import absolute_import, division, print_function

from os import path
import csv

import yaml
from dynd import nd, ndt
import datashape
from datashape.type_equation_solver import matches_datashape_pattern

import blaze
from .. import py2help


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
    ds_str = arrmeta.get('datashape')  # optional. HDF5 does not need that.

    if tp == 'csv':
        with open(fsdir + '.csv', 'r') as f:
            rd = csv.reader(f)
            if imp.get('headers', False):
                # Skip the header line
                next(rd)
            dat = list(rd)
        arr = nd.array(dat, ndt.type(ds_str))[:]
        return blaze.array(arr)
    elif tp == 'json':
        arr = nd.parse_json(ds_str, nd.memmap(fsdir + '.json'))
        return blaze.array(arr)
    elif tp == 'hdf5':
        import tables as tb
        from blaze.datadescriptor import HDF5_DDesc
        fname = fsdir + '.h5'   # XXX .h5 assumed for HDF5
        with tb.open_file(fname, 'r') as f:
            dp = imp.get('datapath')  # specifies a path in HDF5
            try:
                dparr = f.get_node(f.root, dp, 'Leaf')
            except tb.NoSuchNodeError:
                raise RuntimeError(
                    'HDF5 file does not have a dataset in %r' % dp)
            dd = HDF5_DDesc(fname, dp)
        return blaze.array(dd)
    elif tp == 'npy':
        import numpy as np
        use_memmap = imp.get('memmap', False)
        if use_memmap:
            arr = np.load(fsdir + '.npy', 'r')
        else:
            arr = np.load(fsdir + '.npy')
        arr = nd.array(arr)
        arr = blaze.array(arr)
        ds = datashape.dshape(ds_str)
        if not matches_datashape_pattern(arr.dshape, ds):
            raise RuntimeError(('NPY file for blaze catalog path %r ' +
                                'has the wrong datashape (%r instead of ' +
                                '%r)') % (arr.dshape, ds))
        return arr
    elif tp == 'py':
        ds = datashape.dshape(ds_str)
        # The script is run with the following globals,
        # and should put the loaded array in a global
        # called 'result'.
        gbl = {'catconf': conf,  # Catalog configuration object
               'impdata': imp,   # Import data from the .array file
               'catpath': dir,   # Catalog path
               'fspath': fsdir,  # Equivalent filesystem path
               'dshape': ds      # Datashape the result should have
               }
        if py2help.PY2:
            execfile(fsdir + '.py', gbl, gbl)
        else:
            with open(fsdir + '.py') as f:
                code = compile(f.read(), fsdir + '.py', 'exec')
                exec(code, gbl, gbl)
        arr = gbl.get('result', None)
        if arr is None:
            raise RuntimeError(('Script for blaze catalog path %r did not ' +
                                'return anything in "result" variable')
                               % (dir))
        elif not isinstance(arr, blaze.Array):
            raise RuntimeError(('Script for blaze catalog path %r returned ' +
                                'wrong type of object (%r instead of ' +
                                'blaze.Array)') % (type(arr)))
        if not matches_datashape_pattern(arr.dshape, ds):
            raise RuntimeError(('Script for blaze catalog path %r returned ' +
                                'array with wrong datashape (%r instead of ' +
                                '%r)') % (arr.dshape, ds))
        return arr
    else:
        raise ValueError(('Unsupported array type %r from ' +
                          'blaze catalog entry %r')
                         % (tp, dir))

def load_blaze_subcarray(conf, cdir, subcarray):
    import tables as tb
    from blaze.datadescriptor import HDF5_DDesc
    with tb.open_file(cdir.fname, 'r') as f:
        try:
            dparr = f.get_node(f.root, subcarray, 'Leaf')
        except tb.NoSuchNodeError:
            raise RuntimeError(
                'HDF5 file does not have a dataset in %r' % dp)
        dd = HDF5_DDesc(cdir.fname, subcarray)
    return blaze.array(dd)
    
