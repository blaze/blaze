"""
Create scidb kernel implementations.
"""

from __future__ import absolute_import, division, print_function

from blaze.compute.function import function, kernel, elementwise

AFL = 'AFL'
AQL = 'AQL'

#------------------------------------------------------------------------
# Decorators
#------------------------------------------------------------------------

def scidb_function(signature, **metadata):
    """
    Define a SciDB kernel, which must return an AFL query.
    """
    return function(signature, impl=AFL, **metadata)

def scidb_elementwise(signature, **metadata):
    return scidb_function(signature, elementwise=True, **metadata)

#------------------------------------------------------------------------
# Kernel defs
#------------------------------------------------------------------------

def scidb_kernel(blaze_func, kern, signature, **metadata):
    """Define a scidb kernel implementation for the given blaze function"""
    kernel(blaze_func, AFL, kern, signature, **metadata)