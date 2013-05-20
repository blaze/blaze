from __future__ import absolute_import

""" helper functions for common functionality required by different
parts of the APIs"""

from .py3help import basestring
from .datashape import dshape as _dshape_builder

def _normalize_dshape(ds):
    """ In the API, when a datashape is provided we want to support
    them in string form as well. This function will convert from any
    form we want to support in the API inputs into the internal
    datashape object, so the logic is centralized in a single
    place. Any API function that receives a dshape as a parameter
    should convert it using this function """

    return ds if not isinstance(ds, basestring) else _dshape_builder(ds)
