# -*- coding: utf-8 -*-

"""
Create scidb kernel implementations.
"""

from __future__ import print_function, division, absolute_import
from blaze.function import kernel, elementwise

def scidb_kernel(signature, **metadata):
    """
    Define a SciDB kernel, which must return an AFL query.
    """
    return kernel(signature, impl='scidb_afl', **metadata)

def scidb_elementwise(signature, **metadata):
    return scidb_kernel(signature, elementwise=True, **metadata)
