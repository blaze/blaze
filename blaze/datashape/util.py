from __future__ import absolute_import

__all__ = ['dopen', 'dshape', 'cat_dshapes', 'broadcastable',
                'from_ctypes', 'from_cffi']

import operator
import itertools
import ctypes

from . import parser
from .coretypes import DataShape, Fixed, TypeVar, Record, \
                uint8, uint16, uint32, uint64, \
                int8, int16, int32, int64, \
                float32, float64

#------------------------------------------------------------------------
# Utility Functions for DataShapes
#------------------------------------------------------------------------

def dopen(fname):
    contents = open(fname).read()
    return parser.parse_extern(contents)

def dshape(o):
    if isinstance(o, str):
        return parser.parse(o)
    elif isinstance(o, DataShape):
        return o
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

    shapes = [dshape.shape for dshape in dslist]

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

    return outshape

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

def from_ctypes(ctype):
    """
    Constructs a blaze dshape from a ctypes type.
    """
    if issubclass(ctype, ctypes.Structure):
        fields = []
        for nm, tp in ctype._fields_:
            child_ds = from_ctypes(tp)
            fields.append((nm, from_ctypes(tp)))
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
                        'a blaze datashape')
