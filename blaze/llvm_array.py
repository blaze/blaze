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
#    intp shape[nd];
#    intp stride[nd];
# } strided_array(eltype, nd)
#
# # struct {
#    eltype *data;
#    diminfo shape[nd];
# } new_strided_array(eltype, nd)
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
# } new_strided_array_nd(eltype)
#
# struct {
#    eltype *data;
#    int32 nd;
#    intp shape[nd];
#    intp stride[nd];
# } strided_array_nd(eltype)
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
# diminfo shape[nd]; for dimkind = NEW_STRIDED
#
# intp shape[ind];
# intp strides[ind]; dimkind = STRIDED
#

import llvm.core as lc
from llvm.core import Type
import llvm_cbuilder.shortnames as C

# Different Array Types
ARRAYBIT = 1<<4
C_CONTIGUOUS = ARRAYBIT + 0
F_CONTIGUOUS = ARRAYBIT + 1
STRIDED = ARRAYBIT + 2
NEW_STRIDED = ARRAYBIT + 3

HAS_ND = 1<<5
C_CONTIGUOUS_ND = C_CONTIGUOUS + HAS_ND
F_CONTIGUOUS_ND = F_CONTIGUOUS + HAS_ND
STRIDED_ND = STRIDED + HAS_ND
NEW_STRIDED_ND = NEW_STRIDED + HAS_ND

HAS_DIMKIND = 1<<6
C_CONTIGUOUS_DK = C_CONTIGUOUS + HAS_DIMKIND
F_CONTIGUOUS_DK = F_CONTIGUOUS + HAS_DIMKIND
STRIDED_DK = STRIDED + HAS_DIMKIND
NEW_STRIDED_DK = NEW_STRIDED + HAS_DIMKIND

array_kinds = (C_CONTIGUOUS, F_CONTIGUOUS, STRIDED, NEW_STRIDED,
               C_CONTIGUOUS_ND, F_CONTIGUOUS_ND, STRIDED_ND, NEW_STRIDED_ND,
               C_CONTIGUOUS_DK, F_CONTIGUOUS_DK, STRIDED_DK, NEW_STRIDED_DK)

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
int_type = C.int

diminfo_type = Type.struct([intp_type,    # shape
                            intp_type     # stride
                            ], name='diminfo')

zero_p = lc.Constant.int(intp_type, 0)
one_p = lc.Constant.int(intp_type, 1)


_cache = {}
# This is the way we define LLVM arrays.
#  C_CONTIGUOUS, F_CONTIGUOUS, and STRIDED are strongly encouraged...
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
    elif base == NEW_STRIDED:
        dimstr = 'Array_N'
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
    elif base == NEW_STRIDED:
        terms.append(Type.array(diminfo_type, nd))       # diminfo
    elif base == STRIDED:
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
        new_strided = NEW_STRIDED_ND
        c_contiguous = C_CONTIGUOUS_ND
        f_contiguous = F_CONTIGUOUS_ND
    elif a1 == int16_type:
        if arrtyp.element_count < 3 or arrtyp.elements[2] != int16_type:
            return None
        num = 3
        strided = STRIDED_DK
        new_strided = NEW_STRIDED_DK
        c_contiguous = C_CONTIGUOUS_DK
        f_contiguous = F_CONTIGUOUS_DK
    else:
        num = 1
        strided = STRIDED
        new_strided = NEW_STRIDED
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
        return strided, nd, data_type

    if s1.element == diminfo_type:
        return new_strided, nd, data_type
    elif s1.element == intp_type:
        return c_contiguous if c_contig else f_contiguous, nd, data_type
    else:
        return None

# Returns c++ templates for Array_S, Array_F, Array_C, Array_N...
_template_cache = []
def get_cpp_template(typ='all'):
    if len(_template_cache) == 0:
        _make_cpp_templates()
    templates = _template_cache
    if typ in array_kinds:
        indx = array_kinds.index(typ)
        try:
            if (typ & HAS_DIMKIND):
                base = [templates[0], templates[2]]
            elif (typ & HAS_ND):
                base = [templates[0], templates[1]]
            else:
                base = [templates[0]]
            if (typ & (~(HAS_ND | HAS_DIMKIND))) == NEW_STRIDED:
                base.append(templates[3])
        except TypeError:
            base = templates[:4]
        return '\n'.join(base+[templates[indx+4]])
    else:
        return '\n'.join(templates)
    return

# Warning!  This assumes that clang on the system
#   has the same architecture as ctypes...
def _make_cpp_templates():
    global _template_cache
    _template_cache = []
    import ctypes
    plen = ctypes.sizeof(ctypes.c_size_t)
    spaces = ' '*4
    lsize = ctypes.sizeof(ctypes.c_long)
    isize = ctypes.sizeof(ctypes.c_int)
    llsize = ctypes.sizeof(ctypes.c_longlong)
    shsize = ctypes.sizeof(ctypes.c_short)

    if plen == lsize:
        header = "%stypedef long intp;" % spaces
    elif plen == isize:
        header = "%stypdef int intp;" % spaces
    elif plen == llsize:
        header = "%stypedef longlong intp;" % spaces
    else:
        raise ValueError("Size of pointer not recognized.")

    if lsize == 4:
        header2 = "%stypedef long int32;" % spaces
    elif isize == 4:
        header2 = "%stypedef int int32;" % spaces
    else:
        raise ValueError("Cannot find typedef for 32-bit int;")

    if isize == 2:
        header3 = "%stypedef int int16;" % spaces
    elif shsize == 2:
        header3 = "%stypedef short int16;" % spaces

    template_core = """
    template<class T, int ndim>
    struct {name} {{
        T *data;
        {middle}{dims};
    }};
    """

    header4 = """
    template<class T>
    struct diminfo {
        T dim;
        T stride;
    };
    """
    spaces = ' '*8
    middle_map = {'': '',
                  'ND': 'int32 nd;\n%s' % spaces,
                  'DK': 'int16 nd;\n%sint16 dimkind;\n%s' % (spaces, spaces)
                 }

    dims_map = {'F': 'intp dims[ndim]',
                'C': 'intp dims[ndim]',
                'N': 'diminfo<intp> dims[ndim]',
                'S': 'intp dims[ndim];\n%sintp strides[ndim]' % spaces
               }
    templates = [header, header2, header3, header4]
    for end in ['', 'ND', 'DK']:
        for typ in ['C', 'F', 'S', 'N']:
            name = '_'.join(['Array_%s' % typ]+([end] if end else []))
            templates.append(template_core.format(name=name,
                                                  middle=middle_map[end],
                                                  dims=dims_map[typ]))
    _template_cache.extend(templates)
    return

zero_i = lc.Constant.int(int_type, 0)
one_i = lc.Constant.int(int_type, 1)
two_i = lc.Constant.int(int_type, 2)
three_i = lc.Constant.int(int_type, 3)
four_i = lc.Constant.int(int_type, 4)

# Takes pointer to actual array
# load of strides is done at run-time
# indices is a (Python) sequence of llvm integers
def array_pointer(builder, kind, array_ptr, indices):
    data = builder.gep(array_ptr,[zero_p, zero_i])
    base = kind & (~(HAS_ND | HAS_DIMKIND))
    if base == NEW_STRIDED or (kind & HAS_ND) or (kind & HAS_DIMKIND):
        raise ValueError("Unsupported array kind")        
    shapepos = one_i
    stridepos = two_i
    shape = builder.gep(array_ptr, [zero_p, shapepos])
    strides = builder.gep(array_ptr, [zero_p, stridepos])
    nd = shape.type.pointee.count
    loc = Constant.null(intp_type)
    if base in [C_CONTIGUOUS, F_CONTIGUOS]:
        for i in range(nd-1):
            if base == C_CONTIGUOUS:                   
                indx = Constant.int(intp_type, i+1)
            else:
                indx = Constant.int(intp_type, nd-(i+2))
            sval = builder.load(builder.gep(shape, [zero_p, indx]))
            tmp = builder.mul(indices[i], sval)
            loc = builder.add(loc, tmp)
        loc = builder.add(loc, indicies[-1])
        ptr = builder.gep(data, [loc])
    elif base == STRIDED:
        for i in range(nd):
            indx = Constant.int(intp_type, i)
            sval = builder.load(builder.gep(strides,[zero_p, indx]))
            tmp = builder.mul(indices[i], sval)
            loc = builder.add(loc, tmp)
        base = builder.ptrtoint(data, intp_type)
        target = builder.add(base, loc)
        ptr = builder.inttoptr(target, data.type)
    else:
        raise ValueError('Do not  understand array kind')
    return ptr

def array_setitem(builder, kind, array, indices, value):
    ptr = array_pointer(builder, kind, array, indices)
    builder.store(value, ptr)

def array_getitem(builder, kind, array, indices):
    ptr = array_pointer(builder, kind, array, indices)
    val = builder.load(ptr)
    return val

# Similar to a[i,j] or a[:,i,:,j]
# Stack allocate an array of the same type 
# and fill it with the correct positions
# order is a list of llvm integers of Type.int indicating the axes
#    if None, then it is assumed to be equivalent to arange(len(indices)))
# indicies is a list of indices [i,j,...] to extract along the corresponding dimension
def array_getsubarray(builder, array_ptr, indices, order=None):
    kind, nd, data_type = check_array(array_ptr.type.pointee)
    if kind not in [C_CONTIGUOUS, F_CONTIGUOUS, STRIDED]:
        raise ValueError("Not supported for this array-type")
    new_nd = nd - len(indices)
    if new_nd < 0:
        raise ValueError("Two many indices for array shape")
    if new_nd == 0:
        if order is not None:
            raise ValueError("Retrieving single element but order set")
        return array_getitem(builder, kind, array_ptr, indices)
    if order is None:
        order = [Constant.int(int_type, i) for i in range(len(indices))]
    newtype = array_type(new_nd, kind, data_type)
    new = builder.alloca(newtype)
    # Set the data pointer
    # Load the shape array
    shape = builder.gep(array_ptr, [zero_p, one_i])
    builder.store(val)
    # Load the strides array (if necessary)

    return new

def const_intp(builder, value):
    return Constant.int_signextend(intp_type, value)

def ptr_at(builder, ptr, idx):
    return builder.gep(ptr, [auto_const_intp(idx)])

def load_at(builder, ptr, idx):
    return builder.load(ptr_at(ptr, idx))

def store_at(builder, ptr, idx, val):
    builder.store(val, ptr_at(ptr, idx))

def auto_const_intp(v):
    if isinstance(v, (int, long)):
        return const_intp(v)
    else:
        return v


def test():
    arr = array_type(5, C_CONTIGUOUS)
    assert check_array(arr) == (C_CONTIGUOUS, 5, char_type)
    arr = array_type(4, STRIDED)
    assert check_array(arr) == (STRIDED, 4, char_type)
    arr = array_type(3, STRIDED_SOA)
    assert check_array(arr) == (STRIDED_SOA, 3, char_type)

if __name__ == '__main__':
    test()