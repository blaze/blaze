from __future__ import absolute_import
from __future__ import print_function

__all__ = ['KernelDataPrefix', 'UnarySingleOperation', 'UnaryStridedOperation',
        'ExprSingleOperation', 'ExprStridedOperation', 'BinarySinglePredicate',
        'DynamicKernelInstance', 'JITKernelData', 'UnboundCKernelFunction',
        'CKernel', 'wrap_ckernel_func']

import sys
import ctypes

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

class KernelDataPrefix(ctypes.Structure):
    _fields_ = [("function", ctypes.c_void_p),
                    ("destructor", ctypes.CFUNCTYPE(None,
                                    ctypes.c_void_p))]
KernelDataPrefixP = ctypes.POINTER(KernelDataPrefix)
KernelDataPrefixDestructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

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
                ctypes.c_void_p,                   # dst
                ctypes.POINTER(ctypes.c_void_p),   # src
                KernelDataPrefixP)                 # extra
ExprStridedOperation = ctypes.CFUNCTYPE(None,
                ctypes.c_void_p, c_ssize_t,        # dst, dst_stride
                ctypes.POINTER(ctypes.c_void_p),
                        ctypes.POINTER(c_ssize_t), # src, src_stride
                c_ssize_t,                         # count
                KernelDataPrefixP)                 # extra

# Predicates
BinarySinglePredicate = ctypes.CFUNCTYPE(ctypes.c_int, # boolean result
                ctypes.c_void_p,                   # src0
                ctypes.c_void_p,                   # src1
                KernelDataPrefixP)                 # extra

class DynamicKernelInstance(ctypes.Structure):
    _fields_ = [('kernel', ctypes.c_void_p),
                ('kernel_size', ctypes.c_size_t),
                ('free_func', ctypes.CFUNCTYPE(None, ctypes.c_void_p))]

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
        if self._dki and self._dki.kernel:
            # Call the kernel destructor if available
            kp = self.kernel_prefix
            if kp.destructor:
                kp.destructor(self._dki.kernel)
            # Free the kernel data memory and set the pointer to NULL
            self._dki.free_func(self._dki.kernel)
            self._dki.kernel = None

    def __del__(self):
        self.close()

class JITKernelData(ctypes.Structure):
    _fields_ = [('base', KernelDataPrefix),
                ('owner', ctypes.py_object)]

def _jitkerneldata_destructor(jkd_ptr):
    jkd = JITKernelData.from_address(jkd_ptr)
    # Free the reference to the owner object
    _py_decref(jkd.owner)
    jkd.owner = 0
_jitkerneldata_destructor = KernelDataPrefixDestructor(_jitkerneldata_destructor)

def wrap_ckernel_func(func, owner):
    """This function builds a CKernel object around a ctypes
    function pointer, typically created using a JIT like
    Numba or directly using LLVM. The func must have its
    argtypes set, and its last parameter must be a
    KernelDataPrefixP to be a valid CKernel function.
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
    if func.argtypes[-1] != KernelDataPrefixP:
        raise TypeError('The last argument of the ctypes function ' +
                        'pointer must be KernelDataPrefixP')

    # Allocate the memory for the kernel data
    ck = CKernel(functype)
    ksize = ctypes.sizeof(JITKernelData)
    ck.dynamic_kernel_instance.kernel = _malloc(ksize)
    if not ck.dynamic_kernel_instance.kernel:
        raise MemoryError('Failed to allocate CKernel data')
    ck.dynamic_kernel_instance.kernel_size = ksize
    ck.dynamic_kernel_instance.free_func = _free

    # Populate the kernel data with the function
    jkd = JITKernelData.from_address(ck.dynamic_kernel_instance.kernel)
    # Getting the raw pointer address seems to require these acrobatics
    jkd.base.function = ctypes.c_void_p.from_address(ctypes.addressof(func))
    jkd.base.destructor = _jitkerneldata_destructor
    jkd.owner = ctypes.py_object(owner)
    _py_incref(jkd.owner)

    return ck

class UnboundCKernelFunction(object):
    """
    Holds a ckernel function, generally created by a JIT compiler,
    which is for a specified datashape but not bound to particular
    arguments. This allows a JIT compiler to target the ckernel ABI
    without knowing the final data layout by creating the function,
    and later binding the function to the data by filling in the
    kernel data as needed.
    """
    def __init__(self, func_ptr,
                    kernel_data_struct, bind_func, owner):
        """
        Constructs an UnboundCKernelFunction from all its components.

        Parameters
        ----------
        func_ptr : ctypes function pointer
            The function pointer to the ckernel function.
            The function prototype of the kernel is type(func_ptr).
        kernel_data_struct : subclass of ctypes.Structure
            The structure of the kernel data used by the ckernel function.
            Its first member must be called 'base', an instance
            of JITKernelData.
        bind_func : callable
            This function gets called to populate the kernel data
            of a fresh CKernel instance with data from the argument
            data descriptors.
        owner : object
            An object which holds ownership of the function pointer and
            any other data needed to keep the kernel alive.
        """
        # Validate the function pointer
        if not isinstance(func_ptr, ctypes._CFuncPtr):
            raise TypeError('Require a ctypes function pointer to wrap')
        if func_ptr.argtypes is None:
            raise TypeError('The argtypes of the ctypes function ' +
                            'pointer must be set')
        if func_ptr.argtypes[-1] != KernelDataPrefixP:
            raise TypeError('The last argument of the ctypes function ' +
                            'pointer must be KernelDataPrefixP')
        # Validate the kernel data struct
        if kernel_data_struct._fields_[0][0] != 'base':
            raise TypeError('The first field of the kernel data struct ' +
                            'must be called base')
        if kernel_data_struct._fields_[0][1] != JITKernelData:
            raise TypeError('The first field, base, of the kernel data struct ' +
                            'must be a JITKernelData')
        self._func_ptr = func_ptr
        self._kernel_data_struct = kernel_data_struct
        self._bind_func = bind_func
        self._owner = owner

    def bind(self, *args, **kwargs):
        """Creates a new ckernel, and binds the provided data descriptor
        arguments to it by forwarding them to the bind_func.
        """
        # Allocate the memory for the kernel data
        ck = CKernel(type(self.func_ptr))
        ksize = ctypes.sizeof(self.kernel_data_struct)
        ck.dynamic_kernel_instance.kernel = _malloc(ksize)
        if not ck.dynamic_kernel_instance.kernel:
            raise MemoryError('Failed to allocate CKernel data')
        ck.dynamic_kernel_instance.kernel_size = ksize
        ck.dynamic_kernel_instance.free_func = _free

        kd = self.kernel_data_struct.from_address(
                        ck.dynamic_kernel_instance.kernel)
        # Getting the raw pointer address seems to require these acrobatics
        kd.base.base.function = ctypes.c_void_p.from_address(
                        ctypes.addressof(self.func_ptr))
        kd.base.base.destructor = _jitkerneldata_destructor
        kd.base.owner = ctypes.py_object(self.owner)
        _py_incref(kd.base.owner)
        # Call the find function to fill in the rest of the kernel dat
        # TODO: This assumes all the rest of the kernel data is POD,
        #       if it turns out not be POD, we will have to use
        #       a different destructor functions which destroyes
        #       this part of the kernel data as well.
        self.bind_func(kd, *args, **kwargs)
        return ck

    @property
    def func_ptr(self):
        return self._func_ptr

    @property
    def kernel_data_struct(self):
        return self._kernel_data_struct

    @property
    def bind_func(self):
        return self._bind_func

    @property
    def owner(self):
        return self._owner
