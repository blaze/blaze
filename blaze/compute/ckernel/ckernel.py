from __future__ import absolute_import, division, print_function

__all__ = ['JITCKernelData', 'wrap_ckernel_func']

import sys
import ctypes
from dynd import nd, ndt, _lowlevel

from dynd._lowlevel import (CKernelPrefixStruct, CKernelPrefixStructPtr,
        CKernelPrefixDestructor,
        CKernelBuilder,
        UnarySingleOperation, UnaryStridedOperation,
        ExprSingleOperation, ExprStridedOperation, BinarySinglePredicate)

if sys.version_info >= (2, 7):
    c_ssize_t = ctypes.c_ssize_t
else:
    if ctypes.sizeof(ctypes.c_void_p) == 4:
        c_ssize_t = ctypes.c_int32
    else:
        c_ssize_t = ctypes.c_int64

# Get some ctypes function pointers we need
if sys.platform == 'win32':
    _malloc = ctypes.cdll.msvcrt.malloc
    _free = ctypes.cdll.msvcrt.free
else:
    _malloc = ctypes.pythonapi.malloc
    _free = ctypes.pythonapi.free
_malloc.argtypes = (ctypes.c_size_t,)
_malloc.restype = ctypes.c_void_p
_free.argtypes = (ctypes.c_void_p,)
# Convert _free into a CFUNCTYPE so the assignment of it into the struct works
_free_proto = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_free = _free_proto(ctypes.c_void_p.from_address(ctypes.addressof(_free)).value)

_py_decref = ctypes.pythonapi.Py_DecRef
_py_decref.argtypes = (ctypes.py_object,)
_py_incref = ctypes.pythonapi.Py_IncRef
_py_incref.argtypes = (ctypes.py_object,)

class JITCKernelData(ctypes.Structure):
    _fields_ = [('base', CKernelPrefixStruct),
                ('owner', ctypes.py_object)]

def _jitkerneldata_destructor(jkd_ptr):
    jkd = JITCKernelData.from_address(jkd_ptr)
    # Free the reference to the owner object
    _py_decref(jkd.owner)
    jkd.owner = 0
_jitkerneldata_destructor = CKernelPrefixDestructor(_jitkerneldata_destructor)

def wrap_ckernel_func(out_ckb, ckb_offset, func, owner):
    """
    This function generates a ckernel inside a ckernel_builder
    object from a ctypes function pointer, typically created using a JIT like
    Numba or directly using LLVM. The func must have its
    argtypes set, and its last parameter must be a
    CKernelPrefixStructPtr to be a valid CKernel function.
    The owner should be a pointer to an object which
    keeps the function pointer alive.
    """
    functype = type(func)
    # Validate the arguments
    if not isinstance(func, ctypes._CFuncPtr):
        raise TypeError('Require a ctypes function pointer to wrap')
    if func.argtypes is None:
        raise TypeError('The argtypes of the ctypes function ' +
                        'pointer must be set')
    if func.argtypes[-1] != CKernelPrefixStructPtr:
        raise TypeError('The last argument of the ctypes function ' +
                        'pointer must be CKernelPrefixStructPtr')

    # Allocate the memory for the kernel data
    ksize = ctypes.sizeof(JITCKernelData)
    ckb_end_offset = ckb_offset + ksize
    _lowlevel.ckernel_builder_ensure_capacity_leaf(out_ckb, ckb_end_offset)

    # Populate the kernel data with the function
    jkd = JITCKernelData.from_address(out_ckb.data + ckb_offset)
    # Getting the raw pointer address seems to require these acrobatics
    jkd.base.function = ctypes.c_void_p.from_address(ctypes.addressof(func))
    jkd.base.destructor = _jitkerneldata_destructor
    jkd.owner = ctypes.py_object(owner)
    _py_incref(jkd.owner)

    # Return the offset to the end of the ckernel
    return ckb_end_offset

