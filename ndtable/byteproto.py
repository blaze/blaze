import struct
from ctypes import Structure, c_void_p, c_int, sizeof

import llvm.core as lc

#------------------------------------------------------------------------
# Hinting & Flags
#------------------------------------------------------------------------

CONTIGUOUS = 1
STRIDED    = 2
STREAM     = 4
CHUNKED    = 8

READ  = 1
WRITE = 2
READWRITE = READ | WRITE

LOCAL  = 1
REMOTE = 2

ACCESS_ALLOC   = 1
ACCESS_READ    = 2
ACCESS_WRITE   = 4
ACCESS_COPY    = 8
ACCESS_APPEND  = 16

#------------------------------------------------------------------------
# LLVM Primitives
#------------------------------------------------------------------------

#_plat_bits = struct.calcsize('@P') * 8
_plat_bits = 64

_int1      = lc.Type.int(1)
_int8      = lc.Type.int(8)
_int32     = lc.Type.int(32)
_int64     = lc.Type.int(64)

_intp           = lc.Type.int(_plat_bits)
_intp_star      = lc.Type.pointer(_intp)
_void_star      = lc.Type.pointer(lc.Type.int(8))
_void_star_star = lc.Type.pointer(_void_star)

_float      = lc.Type.float()
_double     = lc.Type.double()
_complex64  = lc.Type.struct([_float, _float])
_complex128 = lc.Type.struct([_double, _double])

#------------------------------------------------------------------------
# Buffer
#------------------------------------------------------------------------

_buffer_struct = lc.Type.struct([
    _void_star, # data
    _int8,      # itemsize
])
_buffer_object = lc.Type.pointer(_buffer_struct)
_bufferlist_struct = lambda buf, n: lc.Type.array(buf, n)

#------------------------------------------------------------------------
# Streams
#------------------------------------------------------------------------

# TODO

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def numpy_pointer(numpy_array, ctype=c_void_p):
    return numpy_array.ctypes.data_as(ctype)
