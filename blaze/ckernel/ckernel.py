from __future__ import absolute_import
from __future__ import print_function

__all__ = ['KernelDataPrefix', 'UnarySingleOperation', 'UnaryStridedOperation',
        'ExprSingleOperation', 'ExprStridedOperation', 'BinarySinglePredicate',
        'DynamicKernelInstance', 'DynamicKernelInstanceP', 'CKernel']

import sys
import ctypes

if sys.version_info >= (2, 7):
    c_ssize_t = ctypes.c_ssize_t
else:
    if ctypes.sizeof(ctypes.c_void_p) == 4:
        c_ssize_t = ctypes.c_int32
    else:
        c_ssize_t = ctypes.c_int64

class KernelDataPrefix(ctypes.Structure):
    pass
KernelDataPrefixP = ctypes.POINTER(KernelDataPrefix)
KernelDataPrefix._fields_ = [("function", ctypes.c_void_p),
                ("destructor", ctypes.CFUNCTYPE(None,
                                KernelDataPrefixP))]


# Unary operations (including assignment functions)
UnarySingleOperation = ctypes.CFUNCTYPE(None,
                ctypes.c_void_p,                  # dst
                ctypes.c_void_p,                  # src
                KernelDataPrefixP)                # extra
UnaryStridedOperation = ctypes.CFUNCTYPE(None,
                ctypes.c_void_p, c_ssize_t,      # dst, dst_stride
                ctypes.c_void_p, c_ssize_t,      # src, src_stride
                c_ssize_t,                       # count
                KernelDataPrefixP)               # extra

# Expr operations (array of src operands, how many operands is baked in)
ExprSingleOperation = ctypes.CFUNCTYPE(None,
                ctypes.c_void_p,                  # dst
                ctypes.POINTER(ctypes.c_void_p),  # src
                KernelDataPrefixP)                # extra
ExprStridedOperation = ctypes.CFUNCTYPE(None,
                ctypes.c_void_p, c_ssize_t,        # dst, dst_stride
                ctypes.POINTER(ctypes.c_void_p),
                        ctypes.POINTER(c_ssize_t), # src, src_stride
                c_ssize_t,                         # count
                KernelDataPrefixP)                 # extra

# Predicates
BinarySinglePredicate = ctypes.CFUNCTYPE(ctypes.c_int, # boolean result
                ctypes.c_void_p,                  # src0
                ctypes.c_void_p,                  # src1
                KernelDataPrefixP)                # extra

class DynamicKernelInstance(ctypes.Structure):
    _fields_ = [('kernel', KernelDataPrefixP),
                ('kernel_size', ctypes.c_size_t),
                ('free_func', ctypes.CFUNCTYPE(None,
                        ctypes.c_void_p))]

DynamicKernelInstanceP = ctypes.POINTER(DynamicKernelInstance)

class CKernel(object):
    _dkip = None

    def __init__(self, dkip, kernel_proto):
        """Constructs a CKernel from a raw dynamic_kernel_instance
        pointer, taking ownership of the data inside.

        Parameters
        ----------
        dkip : DynamicKernelInstanceP
            A pointer to a ctypes dynamic_kernel_instance struct.
        kernel_proto : CFUNCPTR
            The function prototype of the kernel.
        """
        if not isinstance(dkip, DynamicKernelInstanceP):
            raise TypeError('CKernel constructor requires a pointer to '
                            'a DynamicKernelInstance structure')
        if not dkip:
            raise ValueError('CKernel constructor requires a non-NULL '
                            'DynamicKernelInstance pointer')
        if not issubclass(kernel_proto, ctypes._CFuncPtr):
            raise ValueError('CKernel constructor requires a ctypes '
                            'function pointer type for kernel_proto')
        self._dkip = dkip
        self._kernel_proto = kernel_proto

    @property
    def dynamic_kernel_instance(self):
        return self._dkip.contents

    @property
    def kernel_function(self):
        return ctypes.cast(self.contents.function, self.kernel_proto)

    @property
    def kernel_prefix(self):
        return self.contents.kernel

    def kernel_proto(self):
        return self._kernel_proto

    def __call__(*args):
        return self.kernel_function(*(args + (self.kernel_prefix,)))

    def close(self):
        if self._dkip and self._dkip.kernel:
            # Call the free function and set the pointer to NULL
            self._dkip.free_func(self._dkip.kernel)
            self._dkip.kernel = KernelDataPrefixP()

    def __del__(self):
        self.close()
