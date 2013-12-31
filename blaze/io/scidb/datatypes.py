"""SciDB type conversions."""

from __future__ import absolute_import, division, print_function

from datashape import coretypes as T


def scidb_measure(measure):
    """Construct a SciDB type from a blaze measure (dtype)"""
    return measure.name  # TODO: HACK, set up a type map


def scidb_dshape(dshape, chunk_size=1024, overlap=0):
    """Construct a SciDB type from a DataShape"""
    import scidbpy
    # TODO: Validate shape regularity
    shape, dtype = T.to_numpy(dshape)
    sdshape = scidbpy.SciDBDataShape(shape, dtype,
                                     chunk_size=chunk_size,
                                     chunk_overlap=overlap)
    return sdshape.schema
