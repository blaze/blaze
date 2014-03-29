from __future__ import absolute_import, division, print_function

#######################################################################
# Checks and variables for optional libraries
#######################################################################

# Check for PyTables
try:
    import tables as tb
    tables_is_here = True
except ImportError:
    tables_is_here = False
