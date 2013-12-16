from __future__ import absolute_import
import os
from os import path
import yaml
import csv
from dynd import nd, ndt
import blaze


def compatible_array_dshape(arr, ds):
    """Checks if the array is compatible with the given dshape.

       Examples
       --------

       >>> compatible_array_dshape(blaze.array([1,2,3]),
       ...                         blaze.dshape("M, int32"))
       True

       >>>
    """
    # To do this check, we unify the array's dshape and the
    # provided dshape. Because the array's dshape is concrete,
    # the result of unification should be equal, otherwise the
    # the unification promoted its type.
    try:
        print(arr.dshape, ds)
        unify_res = blaze.datashape.unify([(arr.dshape, ds)],
                                          broadcasting=[False])
        [unified_ds], constraints = unify_res
    except blaze.error.UnificationError:
        return False

    return unified_ds == arr.dshape


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
    ds_str = arrmeta['datashape']
    ds = blaze.dshape(ds_str)

    if tp == 'csv':
        with open(fsdir + '.csv', 'rb') as f:
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
    elif tp == 'npy':
        import numpy as np
        use_memmap = imp.get('memmap', False)
        if use_memmap:
            arr = np.load(fsdir + '.npy', 'r')
        else:
            arr = np.load(fsdir + '.npy')
        arr = nd.array(arr)
        arr = blaze.array(arr)
        if not compatible_array_dshape(arr, ds):
            raise RuntimeError(('NPY file for blaze catalog path %r ' +
                                'has the wrong datashape (%r instead of ' +
                                '%r)') % (arr.dshape, ds))
        return arr
    elif tp == 'py':
        # The script is run with the following globals,
        # and should put the loaded array in a global
        # called 'result'.
        gbl = {'catconf': conf,  # Catalog configuration object
               'impdata': imp,   # Import data from the .array file
               'catpath': dir,   # Catalog path
               'fspath': fsdir,  # Equivalent filesystem path
               'dshape': ds      # Datashape the result should have
               }
        execfile(fsdir + '.py', gbl)
        arr = gbl.get('result', None)
        if arr is None:
            raise RuntimeError(('Script for blaze catalog path %r did not ' +
                                'return anything in "result" variable')
                               % (dir))
        elif not isinstance(arr, blaze.Array):
            raise RuntimeError(('Script for blaze catalog path %r returned ' +
                                'wrong type of object (%r instead of ' +
                                'blaze.Array)') % (type(arr)))
        if not compatible_array_dshape(arr, ds):
            raise RuntimeError(('Script for blaze catalog path %r returned ' +
                                'array with wrong datashape (%r instead of ' +
                                '%r)') % (arr.dshape, ds))
        return arr
    else:
        raise ValueError(('Unsupported array type %r from ' +
                          'blaze catalog entry %r')
                         % (tp, dir))
