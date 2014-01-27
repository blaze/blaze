"""
Create SQL kernel implementations.
"""

from __future__ import absolute_import, division, print_function

from blaze.compute.function import function, kernel, elementwise

SQL = 'sql'

#------------------------------------------------------------------------
# Decorators
#------------------------------------------------------------------------

def sql_function(signature, **metadata):
    """
    Define an SQL function, which must return an SQL query string.
    """
    return function(signature, impl=SQL, **metadata)

def sql_elementwise(signature, **metadata):
    return sql_function(signature, elementwise=True, **metadata)

#------------------------------------------------------------------------
# Kernel defs
#------------------------------------------------------------------------

def sql_kernel(blaze_func, kern, signature, **metadata):
    """Define an SQL kernel implementation for the given blaze function"""
    kernel(blaze_func, SQL, kern, signature, **metadata)