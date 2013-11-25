# -*- coding: utf-8 -*-

"""
SciDB array constructors.
"""

from __future__ import print_function, division, absolute_import
import blaze

from .query import Query, build
from .datatypes import scidb_dshape
from .datadesc import SciDBDataDesc

import scidbpy

#------------------------------------------------------------------------
# Array creation
#------------------------------------------------------------------------

def _create(dshape, n, chunk_size=1024, overlap=0):
    sdshape = scidb_dshape(dshape, chunk_size, overlap)
    return SciDBDataDesc(build(sdshape, n))

#------------------------------------------------------------------------
# Constructors
#------------------------------------------------------------------------

def empty(dshape, chunk_size=1024, overlap=0):
    return zeros(dshape, chunk_size, overlap)

def zeros(dshape, chunk_size=1024, overlap=0):
    return _create(dshape, "0", chunk_size, overlap)

def ones(dshape, chunk_size=1024, overlap=0):
    return _create(dshape, "1", chunk_size, overlap)