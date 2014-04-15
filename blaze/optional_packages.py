from __future__ import absolute_import, division, print_function

#######################################################################
# Checks and variables for optional libraries
#######################################################################

# Check for PyTables
try:
    import tables
    tables_is_here = True
except ImportError:
    tables_is_here = False

# Check for netcdf4-python
try:
    import netCDF4
    netCDF4_is_here = True
except ImportError:
    netCDF4_is_here = False
