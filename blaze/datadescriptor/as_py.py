from __future__ import absolute_import

from ..datashape import DataShape, CType, Record
from .data_descriptor import IDataDescriptor
import struct
import ctypes

def ctypes_ptr_to_py(ptr_t):
    def ptr_to_py(ptr):
        cptr = ctypes.cast(ptr, ptr_t)
        return cptr.contents.value
    return ptr_to_py

_char_ptr_t = ctypes.POINTER(ctypes.c_uint8)
def bool_to_py(ptr):
    cptr = ctypes.cast(ptr, _char_ptr_t)
    return (cptr.contents.value != 0)

def complex_ptr_to_py(ptr_t, bytes):
    def ptr_to_py(ptr):
        cptr = ctypes.cast(ptr, ptr_t)
        cptr2 = ctypes.cast(ptr+bytes, ptr_t)
        return complex(cptr.contents.value, cptr2.contents.value)
    return ptr_to_py


_dshape_name_to_py = {
    'bool' : bool_to_py,
    'int8' : ctypes_ptr_to_py(ctypes.POINTER(ctypes.c_int8)),
    'int16' : ctypes_ptr_to_py(ctypes.POINTER(ctypes.c_int16)),
    'int32' : ctypes_ptr_to_py(ctypes.POINTER(ctypes.c_int32)),
    'int64' : ctypes_ptr_to_py(ctypes.POINTER(ctypes.c_int64)),
    'uint8' : ctypes_ptr_to_py(ctypes.POINTER(ctypes.c_uint8)),
    'uint16' : ctypes_ptr_to_py(ctypes.POINTER(ctypes.c_uint16)),
    'uint32' : ctypes_ptr_to_py(ctypes.POINTER(ctypes.c_uint32)),
    'uint64' : ctypes_ptr_to_py(ctypes.POINTER(ctypes.c_uint64)),
    'float32' : ctypes_ptr_to_py(ctypes.POINTER(ctypes.c_float)),
    'float64' : ctypes_ptr_to_py(ctypes.POINTER(ctypes.c_double)),
    'complex64': complex_ptr_to_py(ctypes.POINTER(ctypes.c_float), 4),
    'complex128': complex_ptr_to_py(ctypes.POINTER(ctypes.c_double), 8)
}

def dshaped_ptr_to_py(ds):
    """Returns a function which converts pointers to python
    objects for the specified dshape.
    """
    if isinstance(ds, CType):
        # Use the ctypes library to convert these
        ptr_to_py = _dshape_name_to_py.get(ds.name, None)
        if ptr_to_py is None:
            raise TypeError(('Converting data with '
                            '%r to a python object is not yet supported') % (ds))
        return ptr_to_py
    elif isinstance(ds, Record):
        # TODO Add a c_offsets to the Record dshape similar to c_strides
        raise NotImplemented
    else:
        raise TypeError(('Converting data with dshape'
                        '%r to a python object is not yet supported') % (ds))

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
    if len(ds) == 1:
        # Use the get_element interface to get
        # the data as a C pointer
        ptr_to_py = dshaped_ptr_to_py(ds)
        ge = dd.element_reader(0)
        return ptr_to_py(ge.read_single(()))
    elif len(ds) == 2:
        # Use the element_iter interface to get
        # all the elements as C pointers
        ptr_to_py = dshaped_ptr_to_py(ds[-1])
        ei = dd.element_read_iter()
        return [ptr_to_py(ptr) for ptr in ei]
    else:
        # Use the data descriptor iterator to
        # recursively process multi-dimensional arrays
        return [dd_as_py(child_dd) for child_dd in dd]
