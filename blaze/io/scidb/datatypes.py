# -*- coding: utf-8 -*-

"""
SciDB type conversions.
"""

from __future__ import print_function, division, absolute_import
import blaze
from blaze.datashape import coretypes as ds

def scidb_measure(measure):
    """Construct a SciDB type from a blaze measure (dtype)"""
    return measure.name # TODO: HACK, set up a type map

def scidb_dshape(dshape, chunk_size=1024, overlap=0):
    """Construct a SciDB type from a DataShape"""
    import scidbpy
    # TODO: Validate shape regularity
    shape, dtype = ds.to_numpy(dshape)
    sdshape = scidbpy.SciDBDataShape(shape, dtype,
                                     chunk_size=chunk_size,
                                     chunk_overlap=overlap)
    return sdshape.schema
