# This should be moved to llvmpy
# 
# There are different array kinds parameterized by eltype and nd
# 
# Contiguous or Fortran 
# struct {
#    eltype *data;
#    intp shape[nd]; 
# } contiguous_array(eltype, nd)
#
# struct {
#    eltype *data;
#    diminfo shape[nd];
# } strided_array(eltype, nd)
# 
# struct {
#    eltype *data;
#    intp shape[nd];
#    intp stride[nd];
# } strided_soa_array(eltype, nd)
#
# struct {
#   intp dim;
#   intp stride;
#} diminfo
#
# These are for low-level array-routines that need to know the number
# of dimensions at run-time (not just code-generation time):
#
# The first two are recommended
#
# struct {
#   eltype *data;
#   int32 nd;
#   intp shape[nd];
# } contiguous_array_nd(eltype)
#
# struct {
#    eltype *data;
#    int32 nd;
#    diminfo shape[nd];
# } strided_array_nd(eltype)
# 
#
# Backward compatible but deprecated:
# struct {
#    eltype *data;
#    int32 nd;
#    intp shape[nd];
#    intp stride[nd];
# } strided_soa_array_nd(eltype)
#
# 
# The most general (where the kind of array is stored as well as number
#                   of dimensions)
# Rarely needed.
# 
# struct {
#   eltype *data;
#   int16 nd;
#   int16 dimkind;
#   ???
# } array_nd(eltype)
# 
# where ??? is run-time interpreted based on the dimkind to either:
# intp shape[nd];  for dimkind = C_CONTIGUOUS or F_CONTIGUOUS
#
# diminfo shape[nd]; for dimkind = STRIDED
#
# intp shape[ind];
# intp strides[ind]; dimkind = STRIDED_SOA 
#

import llvm.core as lc
from llvm.core import Type
import llvm_cbuilder.shortnames as C

# Different Array Types
ARRAYBIT = 1<<4
C_CONTIGUOUS = ARRAYBIT + 0
F_CONTIGUOUS = ARRAYBIT + 1
STRIDED = ARRAYBIT + 2
STRIDED_SOA = ARRAYBIT + 3

HAS_ND = 1<<5
C_CONTIGUOUS_ND = C_CONTIGUOUS + HAS_ND
F_CONTIGUOUS_ND = F_CONTIGUOUS + HAS_ND
STRIDED_ND = STRIDED + HAS_ND
STRIDED_SOA_ND = STRIDED_SOA + HAS_ND

HAS_DIMKIND = 1<<6
C_CONTIGUOUS_DK = C_CONTIGUOUS + HAS_DIMKIND
F_CONTIGUOUS_DK = F_CONTIGUOUS + HAS_DIMKIND
STRIDED_DK = STRIDED + HAS_DIMKIND
STRIDED_SOA_DK = STRIDED_SOA + HAS_DIMKIND

array_kinds = (C_CONTIGUOUS, F_CONTIGUOUS, STRIDED, STRIDED_SOA,
               C_CONTIGUOUS_ND, F_CONTIGUOUS_ND, STRIDED_ND, STRIDED_SOA_DK, 
               C_CONTIGUOUS_DK, F_CONTIGUOUS_DK, STRIDED_DK, STRIDED_SOA_DK)

_invmap = {}

def kind_to_str(kind):
    global _invmap
    if not _invmap:
        for key, value in globals().items():
            if isinstance(value, int) and value in array_kinds:
                _invmap[value] = key    
    return _invmap[kind]

def str_to_kind(str):
    trial = eval(str)
    if trial not in array_kinds:
        raise ValueError("Invalid Array Kind")
    return trial

void_type = C.void
int32_type = C.int32
char_type = C.char
int16_type = C.int16
intp_type = C.intp

diminfo_type = Type.struct([intp_type,    # shape
                            intp_type     # stride
                            ], name='diminfo')

_cache = {}
# This is the way we define LLVM arrays.
#  CONTIGUOUS and STRIDED are strongly encouraged...
def array_type(nd, kind, el_type=char_type):
    key = (kind, nd, el_type)
    if _cache.has_key(key):
        return _cache[key]

    base = kind & (~(HAS_ND | HAS_DIMKIND))
    if base == C_CONTIGUOUS:
        dimstr = 'Array_C'
    elif base == F_CONTIGUOUS:
        dimstr = 'Array_F'
    elif base == STRIDED:
        dimstr = 'Array_S'
    elif base == STRIDED_SOA:
        dimstr = 'Array_A'
    else:
        raise TypeError("Do not understand Array kind of %d" % kind)

    terms = [Type.pointer(el_type)]        # data

    if (kind & HAS_ND):
        terms.append(int32_type)           # nd
        dimstr += '_ND'
    elif (kind & HAS_DIMKIND):
        terms.extend([int16_type, int16_type]) # nd, dimkind
        dimstr += '_DK'

    if base in [C_CONTIGUOUS, F_CONTIGUOUS]:
        terms.append(Type.array(intp_type, nd))     # shape
    elif base == STRIDED:
        terms.append(Type.array(diminfo_type, nd))       # diminfo
    elif base == STRIDED_SOA: 
        terms.extend([Type.array(intp_type, nd),    # shape
                      Type.array(intp_type, nd)])   # strides

    ret = Type.struct(terms, name=dimstr)
    _cache[key] = ret
    return ret


def check_array(arrtyp):
    if not isinstance(arrtyp, lc.StructType):
        return None
    if arrtyp.element_count not in [2, 3, 4, 5]:
        return None

    # Look through _cache and see if it's there
    for key, value in _cache.items():
        if arrtyp is value:
            return key

    return _raw_check_array(arrtyp)

# Manual check
def _raw_check_array(arrtyp):
    a0 = arrtyp.elements[0]
    a1 = arrtyp.elements[1]
    if not isinstance(a0, lc.PointerType) or \
          not (isinstance(a1, lc.ArrayType) or 
               (a1 == int32_type) or (a1 == int16_type)): 
        return None

    data_type = a0.pointee

    if arrtyp.is_literal:
        c_contig = True
    else:
        if arrtyp.name.startswith('Array_F'):
            c_contig = False
        else:
            c_contig = True


    if a1 == int32_type:
        num = 2
        strided = STRIDED_ND
        strided_soa = STRIDED_SOA_ND
        c_contiguous = C_CONTIGUOUS_ND
        f_contiguous = F_CONTIGUOUS_ND
    elif a1 == int16_type:
        if arrtyp.element_count < 3 or arrtyp.elements[2] != int16_type:
            return None
        num = 3
        strided = STRIDED_DK
        strided_soa = STRIDED_SOA_DK
        c_contiguous = C_CONTIGUOUS_DK
        f_contiguous = F_CONTIGUOUS_DK
    else:
        num = 1       
        strided = STRIDED
        strided_soa = STRIDED_SOA
        c_contiguous = C_CONTIGUOUS
        f_contiguous = F_CONTIGUOUS

    elcount = num + 2
    # otherwise we have lc.ArrType as element [1]
    if arrtyp.element_count not in [num+1,num+2]:
        return None
    s1 = arrtyp.elements[num]
    nd = s1.count

    if arrtyp.element_count == elcount:
        if not isinstance(arrtyp.elements[num+1], lc.ArrayType):
            return None
        s2 = arrtyp.elements[num+1]
        if s1.element != intp_type or s2.element != intp_type:
            return None
        if s1.count != s2.count:
            return None
        return strided_soa, nd, data_type

    if s1.element == diminfo_type:
        return strided, nd, data_type
    elif s1.element == intp_type:
        return c_contiguous if c_contig else f_contiguous, nd, data_type
    else:
        return None


def test():
    arr = array_type(5, C_CONTIGUOUS)
    assert check_array(arr) == (C_CONTIGUOUS, 5, char_type)
    arr = array_type(4, STRIDED)
    assert check_array(arr) == (STRIDED, 4, char_type)
    arr = array_type(3, STRIDED_SOA)
    assert check_array(arr) == (STRIDED_SOA, 3, char_type)

if __name__ == '__main__':
    test()