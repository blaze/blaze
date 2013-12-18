# -*- coding: utf-8 -*-

"""
SciDB array constructors.
"""

from __future__ import print_function, division, absolute_import

import blaze
from blaze.datashape import from_numpy

from .query import Query, build
from .datatypes import scidb_dshape
from .datadescriptor import SciDBDataDescriptor

#------------------------------------------------------------------------
# Array creation
#------------------------------------------------------------------------

def _create(dshape, n, conn, chunk_size=1024, overlap=0):
    sdshape = scidb_dshape(dshape, chunk_size, overlap)
    query = build(sdshape, n)
    return blaze.Array(SciDBDataDescriptor(dshape, query, conn))

#------------------------------------------------------------------------
# Constructors
#------------------------------------------------------------------------

def empty(dshape, conn, chunk_size=1024, overlap=0):
    """Create an empty array"""
    return zeros(dshape, conn, chunk_size, overlap)

def zeros(dshape, conn, chunk_size=1024, overlap=0):
    """Create an array of zeros"""
    return _create(dshape, "0", conn, chunk_size, overlap)

def ones(dshape, conn, chunk_size=1024, overlap=0):
    """Create an array of ones"""
    return _create(dshape, "1", conn, chunk_size, overlap)

def handle(conn, arrname):
    """Obtain an array handle to an existing SciDB array"""
    scidbpy_arr = conn.wrap_array(arrname)
    dshape = from_numpy(scidbpy_arr.shape, scidbpy_arr.dtype)
    return SciDBDataDescriptor(dshape, Query(arrname, ()), conn)
