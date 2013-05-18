from __future__ import absolute_import
from __future__ import print_function

__all__ = ['KernelDataPrefix', 'UnarySingleOperation', 'UnaryStridedOperation',
        'ExprSingleOperation', 'ExprStridedOperation', 'BinarySinglePredicate',
        'DynamicKernelInstance', 'CKernel']

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
    _fields_ = [("function", ctypes.c_void_p),
                    ("destructor", ctypes.CFUNCTYPE(None,
                                    ctypes.c_void_p))]
KernelDataPrefixP = ctypes.POINTER(KernelDataPrefix)

# Unary operations (including assignment functions)
UnarySingleOperation = ctypes.CFUNCTYPE(None,
                ctypes.c_void_p,                  # dst
                ctypes.c_void_p,                  # src
                KernelDataPrefixP)                  # extra
UnaryStridedOperation = ctypes.CFUNCTYPE(None,
                ctypes.c_void_p, c_ssize_t,      # dst, dst_stride
                ctypes.c_void_p, c_ssize_t,      # src, src_stride
                c_ssize_t,                       # count
                KernelDataPrefixP)                 # extra

# Expr operations (array of src operands, how many operands is baked in)
ExprSingleOperation = ctypes.CFUNCTYPE(None,
                ctypes.c_void_p,                  # dst
                ctypes.POINTER(ctypes.c_void_p),  # src
                KernelDataPrefixP)                  # extra
ExprStridedOperation = ctypes.CFUNCTYPE(None,
                ctypes.c_void_p, c_ssize_t,        # dst, dst_stride
                ctypes.POINTER(ctypes.c_void_p),
                        ctypes.POINTER(c_ssize_t), # src, src_stride
                c_ssize_t,                         # count
                KernelDataPrefixP)                   # extra

# Predicates
BinarySinglePredicate = ctypes.CFUNCTYPE(ctypes.c_int, # boolean result
                ctypes.c_void_p,                  # src0
                ctypes.c_void_p,                  # src1
                KernelDataPrefixP)                  # extra

class DynamicKernelInstance(ctypes.Structure):
    _fields_ = [('kernel', ctypes.c_void_p),
                ('kernel_size', ctypes.c_size_t),
                ('free_func', ctypes.CFUNCTYPE(None,
                        ctypes.c_void_p))]

class CKernel(object):
    _dki = None

    def __init__(self, kernel_proto, dki=None):
        """Constructs a CKernel from a raw dynamic_kernel_instance
        pointer, taking ownership of the data inside.

        Parameters
        ----------
        kernel_proto : CFUNCPTR
            The function prototype of the kernel.
        dki : DynamicKernelInstance
            A ctypes dynamic_kernel_instance struct.
        """
        if not issubclass(kernel_proto, ctypes._CFuncPtr):
            raise ValueError('CKernel constructor requires a ctypes '
                            'function pointer type for kernel_proto')
        self._kernel_proto = kernel_proto
        if dki is None:
            self._dki = DynamicKernelInstance()
        else:
            if not isinstance(dki, DynamicKernelInstance):
                raise TypeError('CKernel constructor requires '
                                'a DynamicKernelInstance structure')
            self._dki = dki

    @property
    def dynamic_kernel_instance(self):
        return self._dki

    @property
    def kernel_function(self):
        return ctypes.cast(self.kernel_prefix.function, self.kernel_proto)

    @property
    def kernel_prefix(self):
        return KernelDataPrefix.from_address(self._dki.kernel)

    @property
    def kernel_proto(self):
        return self._kernel_proto

    def __call__(self, *args):
        return self.kernel_function(*(args + (ctypes.byref(self.kernel_prefix),)))

    def close(self):
        if self._dki.kernel:
            # Call the kernel destructor if available
            kp = self.kernel_prefix
            if kp.destructor:
                kp.destructor(self._dki.kernel)
            # Free the kernel data memory and set the pointer to NULL
            self._dki.free_func(self._dki.kernel)
            self._dki.kernel = None

    def __del__(self):
        self.close()
