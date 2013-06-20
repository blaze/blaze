from __future__ import absolute_import

__all__ = ['dopen', 'dshape', 'cat_dshapes', 'broadcastable',
           'from_ctypes', 'from_cffi', 'to_ctypes', 'from_llvm', 'from_blir']

import operator
import itertools
import ctypes
import types
import sys

from . import parser
from .coretypes import (DataShape, Fixed, TypeVar, Record, Wild,
                        uint8, uint16, uint32, uint64, CType,
                        int8, int16, int32, int64,
                        float32, float64, complex64, complex128, Type)

PY3 = (sys.version_info[:2] >= (3,0))

#------------------------------------------------------------------------
# Utility Functions for DataShapes
#------------------------------------------------------------------------

def dopen(fname):
    contents = open(fname).read()
    return parser.parse_extern(contents)

def dshape(o, multi=False):
    if multi:
        return list(parser.parse_mod(o))
    if isinstance(o, str):
        return parser.parse(o)
    elif isinstance(o, DataShape):
        return o
    elif hasattr(o, 'read'):
        return list(parser.parse_mod(o.read()))
    else:
        raise TypeError('Cannot create dshape from object of type %s' % type(o))

def cat_dshapes(dslist):
    """
    Concatenates a list of dshapes together along
    the first axis. Raises an error if there is
    a mismatch along another axis or the measures
    are different.

    Requires that the leading dimension be a known
    size for all data shapes.
    TODO: Relax this restriction to support
          streaming dimensions.
    """
    if len(dslist) == 0:
        raise ValueError('Cannot concatenate an empty list of dshapes')
    elif len(dslist) == 1:
        return dslist[0]

    outer_dim_size = operator.index(dslist[0][0])
    inner_ds = dslist[0][1:]
    for ds in dslist[1:]:
        outer_dim_size += operator.index(ds[0])
        if ds[1:] != inner_ds:
            raise ValueError(('The datashapes to concatenate much'
                            ' all match after'
                            ' the first dimension (%s vs %s)') %
                            (inner_ds, ds[1:]))
    return DataShape([Fixed(outer_dim_size)] + list(inner_ds))


def broadcastable(dslist, ranks=None, rankconnect=[]):
    """Return output (outer) shape if datashapes are broadcastable.

    The default is to assume broadcasting over a scalar operation.
    However, if the kernel to be applied takes arrays as arguments,
    then rank and rank-connect provide the inner-shape information with
    ranks a list of integers indicating the kernel rank of each argument
    and rank-connect a list of sets of tuples where each set contains the
    dimensions that must match and each tuple is (argument #, inner-dim #)
    """
    if ranks is None:
        ranks = [0]*len(dslist)

    shapes = [tuple(operator.index(s) for s in dshape.shape) for dshape in dslist]

    # ensure shapes are large enough
    for i, shape, rank in zip(range(len(dslist)), shapes, ranks):
        if len(shape) < rank:
            raise TypeError(('Argument %d is not large-enough '
                            'for kernel rank') % i)

    splitshapes = [(shape[:len(shape)-rank], shape[len(shape)-rank:])
                             for shape, rank in zip(shapes, ranks)]
    outshapes, inshapes = zip(*splitshapes)

    # broadcast outer-dimensions
    maxshape = max(len(shape) for shape in outshapes)
    outshapes = [(1,)*(maxshape-len(shape))+shape for shape in outshapes]

    # check rank-connections
    for shape1, shape2 in itertools.combinations(outshapes, 2):
        if any((dim1 != 1 and dim2 != 1 and dim1 != dim2)
                  for dim1, dim2 in zip(shape1,shape2)):
            raise TypeError('Outer-dimensions are not broadcastable '
                            'to the same shape')
    outshape = tuple(map(max, zip(*outshapes)))

    for connect in rankconnect:
        for (arg1, dim1), (arg2,dim2) in itertools.combinations(connect, 2):
            if (inshapes[arg1][dim1] != inshapes[arg2][dim2]):
                raise TypeError("Inner dimensions do not match in " +
                                "argument %d and argument %d" % (arg1, arg2))

    return tuple(Fixed(s) for s in outshape)

def _from_cffi_internal(ffi, ctype):
    k = ctype.kind
    if k == 'struct':
        # TODO: Assuming the field offsets match
        #       blaze kernels - need to sync up blaze, dynd,
        #       cffi, numpy, etc so that the field offsets always work!
        #       Also need to make sure there are no bitsize/bitshift
        #       values that would be incompatible.
        return Record([(f[0], _from_cffi_internal(ffi, f[1].type))
                        for f in ctype.fields])
    elif k == 'array':
        if ctype.length is None:
            # Only the first array can have the size
            # unspecified, so only need a single name
            dsparams = [TypeVar('N')]
        else:
            dsparams = [Fixed(ctype.length)]
        ctype = ctype.item
        while ctype.kind == 'array':
            dsparams.append(Fixed(ctype.length))
            ctype = ctype.item
        dsparams.append(_from_cffi_internal(ffi, ctype))
        return DataShape(dsparams)
    elif k == 'primitive':
        cn = ctype.cname
        if cn in ['signed char', 'short', 'int',
                        'long', 'long long']:
            so = ffi.sizeof(ctype)
            if so == 1:
                return int8
            elif so == 2:
                return int16
            elif so == 4:
                return int32
            elif so == 8:
                return int64
            else:
                raise TypeError('cffi primitive "%s" has invalid size %d' %
                                (cn, so))
        elif cn in ['unsigned char', 'unsigned short',
                        'unsigned int', 'unsigned long',
                        'unsigned long long']:
            so = ffi.sizeof(ctype)
            if so == 1:
                return uint8
            elif so == 2:
                return uint16
            elif so == 4:
                return uint32
            elif so == 8:
                return uint64
            else:
                raise TypeError('cffi primitive "%s" has invalid size %d' %
                                (cn, so))
        elif cn == 'float':
            return float32
        elif cn == 'double':
            return float64
        else:
            raise TypeError('Unrecognized cffi primitive "%s"' % cn)
    elif k == 'pointer':
        raise TypeError('a pointer can only be at the outer level of a cffi type '
                        'when converting to blaze datashape')
    else:
        raise TypeError('Unrecognized cffi kind "%s"' % k)


def from_cffi(ffi, ctype):
    """
    Constructs a blaze dshape from a cffi type.
    """
    # Allow one pointer dereference at the outermost level
    if ctype.kind == 'pointer':
        ctype = ctype.item
    return _from_cffi_internal(ffi, ctype)

def to_ctypes(dshape):
    """
    Constructs a ctypes type from a datashape
    """
    if len(dshape) == 1:
        if dshape == int8:
            return ctypes.c_int8
        elif dshape == int16:
            return ctypes.c_int16
        elif dshape == int32:
            return ctypes.c_int32
        elif dshape == int64:
            return ctypes.c_int64
        elif dshape == uint8:
            return ctypes.c_uint8
        elif dshape == uint16:
            return ctypes.c_uint16
        elif dshape == uint32:
            return ctypes.c_uint32
        elif dshape == uint64:
            return ctypes.c_uint64
        elif dshape == float32:
            return ctypes.c_float
        elif dshape == float64:
            return ctypes.c_double
        elif dshape == complex64:
            class Complex64(ctypes.Structure):
                _fields_ = [('real', ctypes.c_float),
                            ('imag', ctypes.c_float)]
                _blaze_type_ = complex64
            return Complex64
        elif dshape == complex128:
            class Complex128(ctypes.Structure):
                _fields_ = [('real', ctypes.c_double),
                            ('imag', ctypes.c_double)]
                _blaze_type_ = complex128
            return Complex128
        elif isinstance(dshape, Record):
            fields = [(name, to_ctypes(dshape.fields[name]))
                                          for name in dshape.names]
            class temp(ctypes.Structure):
                _fields_ = fields
            return temp
        else:
            raise TypeError("Cannot convert datashape %r into ctype" % dshape)
    # Create arrays
    else:
        if isinstance(dshape[0], (TypeVar, Wild)):
            num = 0
        else:
            num = int(dshape[0])
        return num*to_ctypes(dshape.subarray(1))


# FIXME: Add a field
def from_ctypes(ctype):
    """
    Constructs a blaze dshape from a ctypes type.
    """
    if issubclass(ctype, ctypes.Structure):
        fields = []
        if hasattr(ctype, '_blaze_type_'):
            return ctype._blaze_type_
        for nm, tp in ctype._fields_:
            child_ds = from_ctypes(tp)
            fields.append((nm, child_ds))
        ds = Record(fields)
        # TODO: Validate that the ctypes offsets match
        #       the C offsets blaze uses
        return ds
    elif issubclass(ctype, ctypes.Array):
        dstup = []
        while issubclass(ctype, ctypes.Array):
            dstup.append(Fixed(ctype._length_))
            ctype = ctype._type_
        dstup.append(from_ctypes(ctype))
        return DataShape(tuple(dstup))
    elif ctype == ctypes.c_int8:
        return int8
    elif ctype == ctypes.c_int16:
        return int16
    elif ctype == ctypes.c_int32:
        return int32
    elif ctype == ctypes.c_int64:
        return int64
    elif ctype == ctypes.c_uint8:
        return uint8
    elif ctype == ctypes.c_uint16:
        return uint16
    elif ctype == ctypes.c_uint32:
        return uint32
    elif ctype == ctypes.c_uint64:
        return uint64
    elif ctype == ctypes.c_float:
        return float32
    elif ctype == ctypes.c_double:
        return float64
    else:
        raise TypeError('Cannot convert ctypes %r into '
                        'a blaze datashape' % ctype)

# Class to hold Pointer temporarily
def PointerDshape(object):
    def __init__(self, dshape):
        self.dshape = dshape

from ..blaze_kernels import SCALAR, POINTER

def from_llvm(typ, argkind=SCALAR):
    """
    Map an LLVM type to an equivalent datashape type

    argkind is SCALAR, POINTER, or a tuple of (arrkind, nd, el_type) for Arrays
    """
    from ..llvm_array import check_array
    import llvm.core

    kind = typ.kind
    if argkind is None and kind == llvm.core.TYPE_POINTER:
        argkind = check_array(typ.pointee)
        if argkind is None:
            argkind = POINTER
    if kind == llvm.core.TYPE_INTEGER:
        ds = dshape("int" + str(typ.width))

    elif kind == llvm.core.TYPE_DOUBLE:
        ds = float64

    elif kind == llvm.core.TYPE_FLOAT:
        ds = float32

    elif kind == llvm.core.TYPE_VOID:
        ds = None

    elif kind == llvm.core.TYPE_POINTER:
        ds = ''
        pointee = typ.pointee
        p_kind = pointee.kind
        if p_kind == llvm.core.TYPE_INTEGER:
            width = pointee.width
            # Special case:  char * is mapped to strings
            if width == 8:
                ds = dshape("string")
            else:
                ds = PointerDshape(from_llvm(pointee))
        if p_kind == llvm.core.TYPE_STRUCT:
            if argkind == POINTER:
                ds = PointerDshape(from_llvm(pointee))
            else:  # argkind is a tuple of (arrkind, nd, pointer_type)
                nd = argkind[1]
                eltype = from_llvm(argkind[2])
                obj = [TypeVar('i'+str(n)) for n in range(nd)]
                obj.append(eltype)
                ds = DataShape(tuple(obj))
                ds._array_kind = argkind[0]

    elif kind == llvm.core.TYPE_STRUCT:
        if not typ.is_literal:
            struct_name = typ.name.split('.')[-1]
            if not PY3:
                struct_name = struct_name.encode('ascii')
        else:
            struct_name = ''

        names = [ "e"+str(n) for n in range(typ.element_count) ]

        fields = [(name, from_llvm(elem))
                   for name, elem in zip(names, typ.elements)]
        typstr = "{ %s }" % ("; ".join(["{0}: {1}".format(*field)
                                            for field in fields]))

        ds = dshape(typstr)
    else:
        raise TypeError("Unknown type %s" % kind)
    return ds

class TypeSet(object):
    def __init__(self, *args):
        self._set = set(args)
    def __contains__(self, val):
        return val in self._set
    def __repr__(self):
        return "%s()" % (self.__class__.__name__,)

class AnyType(TypeSet):
    def __contains__(self, val):
        return True

class AnyCType(TypeSet):
    def __contains__(self, val):
        return isinstance(val, CType)

    def __str__(self):
        return "*"

def matches_typeset(types, signature):
    match = True
    for a, b in zip(types, signature):
        check = isinstance(b, TypeSet)
        if check and (a not in b) or (not check and a != b):
            match = False
            break
    return match

# FIXME: This is a hack
def from_numba(nty):
    return Type._registry[str(nty)]

def from_numba_str(numba_str):
    import numba
    numba_str = numba_str.strip()
    if numba_str == '*':
        return AnyCType()
    return from_numba(getattr(numba, numba_str))

# Just scalars for now
# FIXME: This could be improved
def to_numba(ds):
    import numba
    return getattr(numba, (str(ds)))

def from_blir(bltype):
    raise NotImplementedError
