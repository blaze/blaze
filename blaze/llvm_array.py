# This should be moved to llvmpy
# 
# We use a simple array type definition at this level for arrays
# 
# struct {
#    eltype *data;
#    int16 nd;
#    int16 kind;   (C_CONTIGUOUS, F_CONTIGUOUS, STRIDED, STRIDED_SOA)
#   union {
#    intp shape[nd];    C_CONTIGUOUS, F_CONTIGUOUS
#    diminfo dims[nd];  STRIDED
#    {intp shape[nd];   STRIDED_SOA
#     intp strides[nd]}
#    }
#} array
#
# struct {
#   intp dim;
#   intp stride;
#} diminfo
#

import llvm.core as lc
from llvm.core import Type, Function, Module



def platform_bytes():
    import sys
    PY3 = (sys.version_info[0] == 3)
    if PY3:
        MAXSIZE = sys.MAXSIZE
    else:
        class X(object):
            def __len__(self):
                return 1 << 32
        try:
            len(X())
        except OverflowError:
            MAXSIZE = int((1 << 31) - 1)
        else:
            MAXSIZE = int((1 << 63) - 1)

    if MAXSIZE > (1 << 32):
        int_bytes = 8
    else:
        int_bytes = 4

    return int_bytes

C_CONTIGUOUS = 0
F_CONTIGUOUS = 1
STRIDED = 2
STRIDED_SOA = 4  # PEP-3118 syle "structure of arrays" shape, stride arrays

void_type = Type.void()
int_type = Type.int()
int16_type = Type.int(2)
intp_type = Type.int(platform_bytes())

diminfo_type = Type.struct([intp_type,    # shape
                            intp_type     # stride
                            ], name='diminfo')

# This is the way we define LLVM arrays. 

def array_type(el_type, nd, kind=C_CONTIGUOUS):
    terms = [Type.pointer(el_type),       # data
             int16_type,                  # nd
             int16_type,                  # diminfo_flag
            ]
    if (kind & STRIDED_SOA):
        terms += [Type.array(intp_type, nd),  # shape[nd]
                  Type.array(intp_type, nd)   # strides[nd]
                  ]
    elif (kind & STRIDED):
        terms += [Type.array(diminfo_type, nd)]  # dims[nd] use 0 for a variable-length struct

    else:
        terms += [Type.array(intp_type, nd)]    # shape[nd]              

    return Type.struct(terms)

generic_array_type = array_type(Type.int(8), 0)


def isarray(arr):
    if not isinstance(arr, lc.StructType):
        return False
    if arr.element_count < 4 or \
        not isinstance(arr.elements[0], lc.PointerType) or \
        not arr.elements[1] == int16_type or \
        not arr.elements[2] == int16_type or \
        not isinstance(arr.elements[3], lc.ArrayType):
        return False
    shapeinfo = arr.elements[3]
    if not (shapeinfo.element == diminfo_type or 
            shapeinfo.element == intp_type):
        return False
    return True

def array_kind(arr):
    pass
