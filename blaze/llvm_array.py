from __future__ import absolute_import

# This should be moved to llvmpy
#
# There are different array kinds parameterized by eltype and nd
#
# Contiguous or Fortran
# struct {
#    eltype *data;
#    intp shape[nd];
#    void *meta;
# } contiguous_array(eltype, nd)
#
# struct {
#    eltype *data;
#    intp shape[nd];
#    intp stride[nd];
#    void *meta;
# } strided_array(eltype, nd)
#
# # struct {
#    eltype *data;
#    diminfo shape[nd];
#    void *meta;
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
#   void *meta;
# } contiguous_array_nd(eltype)
#
# struct {
#    eltype *data;
#    int32 nd;
#    intp shape[nd];
#    intp stride[nd];
#    void *meta;
# } strided_array_nd(eltype)
#
# struct {
#    eltype *data;
#    int32 nd;
#    diminfo shape[nd];
#    void *meta;
# } new_strided_array_nd(eltype)
#
#
#
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
#   void *meta;
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
# Array_C --- C Contiguous
# Array_F --- Fortran Contiguous
# Array_S --- Strided
# Array_CS --- Contiguous in last dimension and strided in others (same layout as Array_S)
# Array_FS --- Contiguous in first dimension and strided in others (same layout as Array_S)
# For 1-d, Array_C, Array_F, Array_CS, and Array_FS behave identically.


import llvm.core as lc
from llvm.core import Type, Constant
import llvm_cbuilder.shortnames as C
from .py3help import reduce

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

# The first three are the most common --- the others are not typically supported
# STRIDED is basically the equivalent of NumPy
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
char_p_type = lc.Type.pointer(C.char)
void_p_type = C.void_p

diminfo_type = Type.struct([intp_type,    # shape
                            intp_type     # stride
                            ], name='diminfo')

zero_p = lc.Constant.int(intp_type, 0)
one_p = lc.Constant.int(intp_type, 1)

# We use a per-module cache because the LLVM linker wants a new struct
#   with the same name in different modules.
# The linker does *not* like the *same* struct with the *same* name in
#   two different modules.
_cache = {}
# This is the way we define LLVM arrays.
#  C_CONTIGUOUS, F_CONTIGUOUS, and STRIDED are strongly encouraged...
def array_type(nd, kind, el_type=char_type, module=None):
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

    if (kind & HAS_ND):
        dimstr += '_ND'
    elif (kind & HAS_DIMKIND):
        dimstr += '_DK'

    key = "%s_%s_%d" % (dimstr, str(el_type), nd)
    if module is not None:
        modcache = _cache.setdefault(module.id,{})
        if key in modcache:
            return modcache[key]

    terms = [Type.pointer(el_type)]        # data

    if (kind & HAS_ND):
        terms.append(int32_type)           # nd
    elif (kind & HAS_DIMKIND):
        terms.extend([int16_type, int16_type]) # nd, dimkind

    if base in [C_CONTIGUOUS, F_CONTIGUOUS]:
        terms.append(Type.array(intp_type, nd))     # shape
    elif base == NEW_STRIDED:
        terms.append(Type.array(diminfo_type, nd))       # diminfo
    elif base == STRIDED:
        terms.extend([Type.array(intp_type, nd),    # shape
                      Type.array(intp_type, nd)])   # strides

    terms.append(void_p_type)
    ret = Type.struct(terms, name=key)
    if module is not None:
        modcache[key] = ret
    return ret

def check_array(arrtyp):
    """Converts an LLVM type into an llvm_array 'kind' for a
    blaze kernel to use.

    Parameters
    ----------
    arrtyp : LLVM type
        The LLVM type to convert into a 'kind'. This type should
        have been created with the array_type function.

    Returns
    -------
    None if the input parameter is not an array_type instance,
    or a 3-tuple (array_kind, ndim, llvm_eltype). The array_kind
    is an integer flags containing values like C_CONTIGUOUS, HAS_ND,
    etc.
    """
    if not isinstance(arrtyp, lc.StructType):
        return None
    if arrtyp.element_count not in [3, 4, 5, 6]:
        return None

    # Look through _cache and see if it's there
    for key, value in _cache.items():
        if arrtyp is value:
            return key[0], key[1], value.elements[0].pointee

    return _raw_check_array(arrtyp)

# Manual check
def _raw_check_array(arrtyp):
    a0 = arrtyp.elements[0]
    a1 = arrtyp.elements[1]
    if not isinstance(a0, lc.PointerType) or \
          not (isinstance(a1, lc.ArrayType) or
               (a1 == int32_type) or (a1 == int16_type)):
        return None

    if not (arrtyp.elements[-1] == void_p_type):
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

    # otherwise we have lc.ArrType as element [1]
    if arrtyp.element_count not in [num+2,num+3]:
        return None
    s1 = arrtyp.elements[num]
    nd = s1.count

    # Strided case
    if arrtyp.element_count == num+3:
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
        void *meta;
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

def const_intp(value):
    return Constant.int_signextend(intp_type, value)

def ptr_at(builder, ptr, idx):
    return builder.gep(ptr, [auto_const_intp(idx)])

def load_at(builder, ptr, idx):
    return builder.load(ptr_at(builder, ptr, idx))

def store_at(builder, ptr, idx, val):
    builder.store(val, ptr_at(builder, ptr, idx))

def get_data_ptr(builder, arrptr):
    val = builder.gep(arrptr, [zero_p, zero_i])
    return builder.load(val)

def get_shape_ptr(builder, arrptr):
    val = builder.gep(arrptr, [zero_p, one_i, zero_p])
    return val

def get_strides_ptr(builder, arrptr):
    return builder.gep(arrptr, [zero_p, two_i, zero_p])

def auto_const_intp(v):
    if hasattr(v, '__index__'):
        return const_intp(v.__index__())
    else:
        return v

# Assumes that unpacked structures with elements all of width > 32bits
# are the same as packed structures --- possibly not true on every platform.
def _sizeof(_eltype, unpacked=False):
    msg = "Cannot determine size of unpacked structure with elements of size %d"
    kind = _eltype.kind
    if kind == lc.TYPE_INTEGER:
        width = _eltype.width
        if width % 8 != 0:
            raise ValueError("Invalid bit-width on Integer")
        if unpacked and width < 32:
            raise ValueError(msg % width)
        return width >> 3
    elif kind == lc.TYPE_POINTER:
        return intp_type.width >> 3
    elif kind == lc.TYPE_FLOAT:
        return 4
    elif kind == lc.TYPE_DOUBLE:
        return 8
    elif kind == lc.TYPE_HALF:
        if unpacked:
            raise ValueError(msg % 2)
        return 2
    elif kind == lc.TYPE_FP128:
        return 16
    elif kind == lc.TYPE_ARRAY:
        return _eltype.count * _sizeof(_eltype.element)
    elif kind == lc.TYPE_STRUCT:
        return sum(_sizeof(element, not _eltype.packed)
                       for element in _eltype.elements)
    raise ValueError("Unimplemented type % s (kind=%s)" % (_eltype, kind))

orderchar = {C_CONTIGUOUS:'C',
             F_CONTIGUOUS:'F',
             STRIDED: 'S'}

kindfromchar = {}
for key, value in orderchar.items():
    kindfromchar[value] = key

# Return the sizeof an LLVM-Type as a runtime value
# This will become a constant at run-time...
def sizeof(llvm_type, builder):
    nullval = Constant.null(Type.pointer(llvm_type))
    size = builder.gep(nullval, [one_i])
    sizeI = builder.bitcast(size, int_type)
    return sizeI

# Return the byte offset of a fieldnumber in a struct
# This will become a constant at run-time...
def offsetof(struct_type, fieldnum, builder):
    nullval = Constant.null(Type.pointer(struct_type))
    if hasattr(fieldnum, '__index__'):
        fieldnum = fieldnum.__index__()
        fieldnum = Constant.int(int_type, fieldnum)
    offset = builder.gep(nullval, [zero_p, fieldnum])
    offsetI = builder.bitcast(offset, int_type)
    return offsetI

LLVM_SCALAR = [lc.TYPE_HALF, lc.TYPE_FP128,
               lc.TYPE_DOUBLE, lc.TYPE_INTEGER, lc.TYPE_FLOAT]
# Embed an llvmvalue into an Array_S array with dimension nd
# by setting strides of new dimensions to 0.
# return the embedded array as an LLArray
# preload the shape and strides of the new array if preload
def embed(builder, llvmval, nd, preload=True):
    raise NotImplementedError
    kind = llvmval.type.kind
    if kind in LLVM_SCALAR:
        pass
    if preload:
        builder_val = builder
    else:
        builder_val = None
    return LLArray(new_ptr, builder_val)

def isinteger(val):
    if hasattr(val, '__index__'):
        return True
    if isinstance(val, lc.Value) and val.type.kind == lc.TYPE_INTEGER:
        return True
    return False

def isiterable(val):
    import collections
    return isinstance(val, collections.Iterable)

# An Array object wrapper around pointer to low-level LLVM array.
# allows pre-loading of strides, shape, so that
# slicing and element-access computation happens at Python level
# without un-necessary memory-access (loading)
# If builder is set during creation, then pre-load
#  Otherwise, attach builder later as attribute
class LLArray(object):
    _strides_ptr = None
    _strides = None
    _shape_ptr = None
    _shape = None
    _data_ptr = None
    _freefuncs = []
    _freedata = []
    _builder_msg = "The builder attribute is not set."
    def __init__(self, array_ptr, builder=None):
        self.builder = builder
        self.array_ptr = array_ptr
        self.array_type = array_ptr.type.pointee
        kind, nd, eltype = check_array(self.array_type)
        self._kind = kind
        self.nd = nd
        self._eltype = eltype
        try:
            self._order = orderchar[kind]
        except KeyError:
            raise ValueError("Unsupported array type %s" % kind)
        self._itemsize = _sizeof(eltype)
        if builder is not None:
            _ = self.strides  # execute property codes to pre-load
            _ = self.shape
            _ = self.data

    @property
    def strides(self):
        if self._kind != STRIDED:
            return None
        if not self._strides:
            if self.builder is None:
                raise ValueError(self._builder_msg)
            if self.nd > 0:
                self._strides_ptr = get_strides_ptr(self.builder, self.array_ptr)
            self._strides = self.preload(self._strides_ptr)
        return self._strides

    @property
    def shape(self):
        if not self._shape:
            if self.builder is None:
                raise ValueError(self._builder_msg)
            if self.nd > 0:
                self._shape_ptr = get_shape_ptr(self.builder, self.array_ptr)
            self._shape = self.preload(self._shape_ptr)
        return self._shape

    @property
    def data(self):
        if not self._data_ptr:
            if self.builder is None:
                raise ValueError(self._builder_msg)
            self._data_ptr = get_data_ptr(self.builder, self.array_ptr)
        return self._data_ptr

    @property
    def itemsize(self):
        if self.builder:
            return sizeof(self._eltype, self.builder)
        else:
            return Constant.int(int_type, self._itemsize)

    @property
    def module(self):
        return self.builder.basic_block.function.module if self.builder else None

    def preload(self, llarray_ptr, count=None):
        if llarray_ptr is None:
            return None
        if count is None:
            count = self.nd
        return [load_at(self.builder, llarray_ptr, i) for i in range(count)]

    def getview(self, nd=None, kind=None, eltype=None):
        newtype = array_type(self.nd if nd is None else nd,
                             self._kind if kind is None else kind,
                             self._eltype if eltype is None else eltype,
                             self.module)
        new = self.builder.alloca(newtype)
        return LLArray(new)

    def getptr(self, *indices):
        assert len(indices) == self.nd
        indices = [auto_const_intp(x) for x in indices]
        shape = self.shape
        strides = self.strides
        order = self._order
        data = self._data_ptr
        builder = self.builder
        intp = intp_type
        if self.nd == 0:
            ptr = builder.gep(data, [zero_p])
        elif order in 'CF':
            # optimize for C and F contiguous
            if order == 'F':
                shape = list(reversed(shape))
            loc = Constant.null(intp)
            for ival, sval in zip(indices, shape[1:]):
                tmp = builder.mul(ival, sval)
                loc = builder.add(loc, tmp)
            loc = builder.add(loc, indices[-1])
            ptr = builder.gep(data, [loc])
        else:
            # any order
            loc = Constant.null(intp)
            for i, s in zip(indices, strides):
                tmp = builder.mul(i, s)
                loc = builder.add(loc, tmp)
            base = builder.ptrtoint(data, intp)
            target = builder.add(base, loc)
            ptr = builder.inttoptr(target, data.type)
        return ptr

    #Can be used to get subarrays as well as elements
    def __getitem__(self, key):
        from .llgetitem import from_Array
        isiter = isiterable(key)
        char = orderchar[self._kind]
        # full-indexing
        if (isiter and len(key) == self.nd) or \
           (self.nd == 1 and isinteger(key)):
            if isiter:
                if any(x in [Ellipsis, None] for x in key):
                    return from_Array(self, key, char)
                else:
                    args = key
            else:
                args = (key.__index__(),)
            ptr = self.getptr(*args)
            return self.builder.load(ptr)
        elif self._kind in [C_CONTIGUOUS, F_CONTIGUOUS, STRIDED]:
            return from_Array(self, key, char)
        else:
            raise NotImplementedError

    # Could use memcopy and memmove to implement full slicing capability
    # But for now just works for single element
    def __setitem__(self, key, value):
        if not isinstance(key, tuple) and len(key) != self.nd:
            raise NotImplementedError
        ptr = self.getptr(*key)
        self.builder.store(value, ptr)

    # Static alloca the structure and malloc the data
    # There is no facility for ensuring lifetime of the memory
    # So this array should *not* be used in another thread
    # shape is a Python list of integers or llvm ints
    # eltype is an llvm type
    # This is intended for temporary use only.
    def create(self, shape=None, kind=None, eltype=None, malloc=None, free=None, order='K'):
        res =  create_array(self.builder, shape or self.shape,
                                          kind or self._kind,
                                          eltype or self._eltype,
                                          malloc, free, order)
        new, freefuncs, char_data = res
        self._freefuncs.append(free)
        self._freedata.append(char_data)
        return LLArray(new)

    def _dealloc(self):
        for freefunc, freedatum in zip(self._freefuncs, self._freedata):
            self.builder.call(freefunc, [freedatum])
        self._freefuncs = []
        self._freedata = []

# Static alloca the structure and malloc the data
# There is no facility for ensuring lifetime of the memory
# So this array should *not* be used in another thread
# shape is a Python list of integers or llvm ints
# eltype is an llvm type
# This is intended for temporary use only.
def create_array(builder, shape, kind, eltype, malloc=None, free=None, order='K'):
    import operator
    if malloc is None:
        malloc, free = _default_malloc_free(builder.basic_block.function.module)
    nd = len(shape)
    newtype = array_type(nd, kind, eltype, builder.basic_block.function.module)
    new = builder.alloca(newtype)
    shape_ptr = get_shape_ptr(builder, new)

    # Store shape
    for i, val in enumerate(shape):
        store_at(builder, shape_ptr, i, auto_const_intp(val))

    # if shape is all integers then we can pre-multiply to get size.
    # Otherwise, we will have to compute the size in the code.
    if all(hasattr(x, '__index__') for x in shape):
        shape = [x.__index__() for x in shape]
        total = reduce(operator.mul, shape, _sizeof(eltype))
        arg = Constant.int(intp_type, total)
        precompute=True
    else:
        precompute=False
        result = sizeof(eltype, builder)
        for val in shape:
            result = builder.mul(result, auto_const_intp(val))
        arg = result
    char_data = builder.call(malloc, [arg])
    data = builder.bitcast(char_data, Type.pointer(eltype))
    data_ptr = builder.gep(new, [zero_p, zero_i])
    builder.store(data, data_ptr)

    if kind == STRIDED:
        # Fill the strides array depending on order
        if order == 'K':
            # if it's strided, I suppose we should choose based on which is
            # larger in self, the first or last stride for now just 'C'
            order = 'F' if kind is F_CONTIGUOUS else 'C'
        if order == 'C':
            range2 = range(nd-1, -1, -1)
            func = operator.sub
        elif order == 'F':
            range2 = range(0, nd, 1)
            func = operator.add
        else:
            raise ValueError("Invalid order given")
        range1 = range2[:-1]
        range3 = [func(v, 1) for v in range1]
        strides_ptr = get_strides_ptr(builder, new)
        if precompute:
            strides_list = [sizeof(eltype)]
            value = strides_list[0]
            for index in range1:
                value = value * shape[index]
                strides_list.append(value)
            for stride, index in zip(strides_list, range2):
                sval = Constant.int(intp_type, stride)
                store_at(builder, strides_ptr, index, sval)
        else:
            sval = Constant.int(intp_type, sizeof(eltype))
            store_at(builder, strides_ptr, range1[0], sval)
            for sh_index, st_index in (range1, range3):
                val = load_at(builder, shape_ptr, sh_index)
                sval = builder.mul(sval, val)
                store_at(builder, strides_ptr, st_index, sval)

    return new, free, char_data


malloc_sig = lc.Type.function(char_p_type, [intp_type])
free_sig = lc.Type.function(void_type, [char_p_type])

def _default_malloc_free(mod):
    malloc = mod.get_or_insert_function(malloc_sig, 'malloc')
    free = mod.get_or_insert_function(free_sig, 'free')
    return malloc, free

def test():
    arr = array_type(5, C_CONTIGUOUS)
    assert check_array(arr) == (C_CONTIGUOUS, 5, char_type)
    arr = array_type(4, STRIDED)
    assert check_array(arr) == (STRIDED, 4, char_type)
    arr = array_type(3, NEW_STRIDED)
    assert check_array(arr) == (NEW_STRIDED, 3, char_type)

if __name__ == '__main__':
    test()