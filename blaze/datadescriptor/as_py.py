from __future__ import absolute_import

from ..datashape import DataShape, CType, Record
from ..py2help import izip
from .data_descriptor import IDataDescriptor
import struct
import ctypes
from dynd import nd, ndt

def dd_as_py(dd):
    """
    Converts the data in a data descriptor into Python
    types. This uses the data_descriptor iteration methods,
    so is not expected to be fast. Its main initial purpose
    is to assist with writing unit tests.
    """
    if not isinstance(dd, IDataDescriptor):
        raise TypeError('expected DataDescriptor, got %r' % type(dd))
    ds = dd.dshape
    if dd.is_concrete:
        return nd.as_py(dd.dynd_arr())
    else:
        # Use the data descriptor iterator to
        # recursively process multi-dimensional arrays
        return [dd_as_py(child_dd) for child_dd in dd]
