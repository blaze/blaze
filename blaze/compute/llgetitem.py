from .llvm_array import (array_type, const_intp, auto_const_intp, 
                         intp_type, int_type,
                         store_at, load_at, get_shape_ptr, get_data_ptr,
                         get_strides_ptr, sizeof, isinteger, isiterable,
                         F_CONTIGUOUS, C_CONTIGUOUS, STRIDED)
from llvm.core import Constant, Type
import llvm.core as lc
import itertools


def _check_N(N):
    if N is None:
        raise ValueError("negative integers not supported")


def adjust_slice(key, N=None):
    start = key.start
    if start is None:
        start = 0

    if start < 0:
        _check_N(N)        
        while start < 0:
            start += N

    stop = key.stop
    if stop is None:
        _check_N(N)
        stop = N

    if stop < 0:
        _check_N(N)
        while stop < 0:
            stop += N

    step = key.step
    if step is None:
        step = 1

    return start, stop, step


# STRIDED

def Sarr_from_S(arr, key):
    raise NotImplementedError


def Sarr_from_S_slice(arr, start, stop, step):
    raise NotImplementedError


def from_S_int(arr, index):
    return from_S_ints(arr, (index,))

def from_S_ints(arr, key):
    raise NotImplementedError
    builder = arr.builder
    num = len(key)
    newnd = arr.nd - num
    if newnd < 0:
        raise ValueError("Too many keys")
    new = arr.getview(nd=newnd)

    oldshape = get_shape_ptr(builder, arr.array_ptr)
    newshape = get_shape_ptr(builder, new.array_ptr)
    # Load the shape array
    for i in range(newnd):
        val = load_at(builder, oldshape, i+num)
        store_at(builder, newshape, i, val)

    # Load the data-pointer
    old_data_ptr = get_data_ptr(builder, arr.array_ptr)
    new_data_ptr = get_data_ptr(builder, new.array_ptr)

    loc = Constant.int(intp_type, 0)
    factor = Constant.int(intp_type, 1)
    for index in range(arr.nd-1,-1,-1):
        val = load_at(builder, oldshape, index)
        factor = builder.mul(factor, val)
        if index < num: # 
            keyval = auto_const_intp(key[index])
            # Multiply by strides
            tmp = builder.mul(keyval, factor)
            # Add to location
            loc = builder.add(loc, tmp)
    ptr = builder.gep(old_data_ptr, [loc])
    builder.store(ptr, new_data_ptr)
    return new

def from_S_slice(arr, start, end):
    raise NotImplementedError
    builder = arr.builder
    new = arr.getview()
    # Load the shape array
    oldshape = get_shape_ptr(builder, arr.array_ptr)
    newshape = get_shape_ptr(builder, new.array_ptr)
    diff = Constant.int(intp_int, end-start)
    store_at(builder, newshape, 0, diff)
    for i in range(1, new.nd):
        val = load_at(builder, oldshape, i)
        store_at(builder, newshape, i, val)

    # Data Pointer
    old_data_ptr = get_data_ptr(builder, arr.array_ptr)
    loc = Constant.int(intp_type, start)
    while dim in arr.shape[1:]:
        loc = builder.mul(loc, dim)
    ptr = builder.gep(old_data_ptr, [loc])
    new_data_ptr = get_data_ptr(builder, new.array_ptr)
    builder.store(ptr, new_data_ptr)
    return new

# FORTRAN CONTIGUOUS

def Sarr_from_F(arr, key):
    raise NotImplementedError

def Sarr_from_F_slice(arr, start, stop, step):
    raise NotImplementedError

def from_F_int(arr, index):
    return from_F_ints(arr, (index,))

# key will be *just* the final integers to extract
#  so that resulting array stays F_CONTIGUOUS
def from_F_ints(arr, key):
    raise NotImplementedError
    builder = arr.builder
    num = len(key)
    newnd = arr.nd - num
    if newnd < 0:
        raise ValueError("Too many keys")
    new = arr.getview(nd=newnd)

    oldshape = get_shape_ptr(builder, arr.array_ptr)
    newshape = get_shape_ptr(builder, new.array_ptr)
    # Load the shape array
    for i in range(newnd):
        val = load_at(builder, oldshape, i+num)
        store_at(builder, newshape, i, val)

    # Load the data-pointer
    old_data_ptr = get_data_ptr(builder, arr.array_ptr)
    new_data_ptr = get_data_ptr(builder, new.array_ptr)

    loc = Constant.int(intp_type, 0)
    factor = Constant.int(intp_type, 1)
    for index in range(arr.nd-1,-1,-1):
        val = load_at(builder, oldshape, index)
        factor = builder.mul(factor, val)
        if index < num: # 
            keyval = auto_const_intp(key[index])
            # Multiply by strides
            tmp = builder.mul(keyval, factor)
            # Add to location
            loc = builder.add(loc, tmp)
    ptr = builder.gep(old_data_ptr, [loc])
    builder.store(ptr, new_data_ptr)
    return new

def from_F_slice(arr, start, end):
    raise NotImplementedError
    builder = arr.builder
    new = arr.getview()
    # Load the shape array
    oldshape = get_shape_ptr(builder, arr.array_ptr)
    newshape = get_shape_ptr(builder, new.array_ptr)
    diff = Constant.int(intp_int, end-start)
    store_at(builder, newshape, 0, diff)
    for i in range(1, new.nd):
        val = load_at(builder, oldshape, i)
        store_at(builder, newshape, i, val)

    # Data Pointer
    old_data_ptr = get_data_ptr(builder, arr.array_ptr)
    loc = Constant.int(intp_type, start)
    while dim in arr.shape[1:]:
        loc = builder.mul(loc, dim)
    ptr = builder.gep(old_data_ptr, [loc])
    new_data_ptr = get_data_ptr(builder, new.array_ptr)
    builder.store(ptr, new_data_ptr)
    return new


# C-CONTIGUOUS

def Sarr_from_C(arr, key):
    raise NotImplementedError

def Sarr_from_C_slice(arr, start, stop, step):
    builder = arr.builder
    new = arr.getview(kind=STRIDED)
    oldshape = get_shape_ptr(builder, arr.array_ptr)
    newshape = get_shape_ptr(builder, new.array_ptr)
    newstrides = get_strides_ptr(bulder, new.array_ptr)

    if all(hasattr(x, '__index__') for x in [start, stop, step]):
        step = auto_const_intp(step)
        newdim = auto_const_intp((stop - start) // step)
    else:
        start, stop, step = [auto_const_intp(x) for x in [start, stop, step]]
        tmp = builder.sub(stop, start)
        newdim = builder.udiv(tmp, step)

    store_at(builder, newshape, 0, newdim)
    # Copy other dimensions over
    for i in range(1, arr.nd):
        val = load_at(builder, oldshape, i)
        store_at(builder, newshape, i, val)

    raise NotImplementedError
    # Fill-in strides
    # Update data-ptr

def from_C_int(arr, index):
    return from_C_ints(arr, (index,))

def from_C_ints(arr, key):
    builder = arr.builder
    num = len(key)
    newnd = arr.nd - num
    if newnd < 0:
        raise ValueError("Too many keys")
    new = arr.getview(nd=newnd)

    oldshape = get_shape_ptr(builder, arr.array_ptr)
    newshape = get_shape_ptr(builder, new.array_ptr)
    # Load the shape array
    for i in range(newnd):
        val = load_at(builder, oldshape, i+num)
        store_at(builder, newshape, i, val)

    # Load the data-pointer
    old_data_ptr = get_data_ptr(builder, arr.array_ptr)
    new_data_ptr = get_data_ptr(builder, new.array_ptr)

    loc = Constant.int(intp_type, 0)
    factor = Constant.int(intp_type, 1)
    for index in range(arr.nd-1,-1,-1):
        val = load_at(builder, oldshape, index)
        factor = builder.mul(factor, val)
        if index < num: # 
            keyval = auto_const_intp(key[index])
            # Multiply by strides
            tmp = builder.mul(keyval, factor)
            # Add to location
            loc = builder.add(loc, tmp)
    ptr = builder.gep(old_data_ptr, [loc])
    builder.store(ptr, new_data_ptr)
    return new

def from_C_slice(arr, start, end):
    builder = arr.builder
    new = arr.getview()
    # Load the shape array
    oldshape = get_shape_ptr(builder, arr.array_ptr)
    newshape = get_shape_ptr(builder, new.array_ptr)
    diff = Constant.int(intp_int, end-start)
    store_at(builder, newshape, 0, diff)
    for i in range(1, new.nd):
        val = load_at(builder, oldshape, i)
        store_at(builder, newshape, i, val)

    # Data Pointer
    old_data_ptr = get_data_ptr(builder, arr.array_ptr)
    loc = Constant.int(intp_type, start)
    while dim in arr.shape[1:]:
        loc = builder.mul(loc, dim)
    ptr = builder.gep(old_data_ptr, [loc])
    new_data_ptr = get_data_ptr(builder, new.array_ptr)
    builder.store(ptr, new_data_ptr)
    return new


# get just the integers
def _convert(x):
    if hasattr(x, '__index__'):
        return x.__index__()
    else:
        return x
    
_keymsg = "Unsupported getitem value %s"
# val is either Ellipsis or slice object.
# check to see if start, stop, and/or step is given for slice
def _needstride(val):
    if not isinstance(val, slice):
        return False
    if val.start is not None and val.start != 0:
        return True
    if val.stop is not None:
        return True
    if (val.step is not None) and (val.step != 1):
        return True
    return False


def _getitem_C(arr, key):
    lastint = None
    needstrided = False
    # determine if 1) the elements of the geitem iterable are
    #                 integers (LLVM or Python indexable), Ellipsis,
    #                 or slice objects
    #              2) the integer elements are all at the front
    #                 so that the resulting slice is continuous
    for i, val in enumerate(key):
        if isinteger(val):
            if lastint is not None:
                needstrided = True
        elif isinstance(val, (Ellipsis, slice)):
            if lastint is None:
                lastint = i
            needstrided = _needstride(val)
        else:
            raise ValueError(_keymsg % val)

    if not needstrided:
        key = [_convert(x) for x in itertools.islice(key, lastint)]

    return needstrided, key


def _getitem_F(arr, key):
    # This looks for integers at the end of the key iterable
    # arr[:,...,i,j] would not need strided
    # arr[:,i,:,j] would need strided as would a[:,i,5:20,j]
    #      and a[:,...,5:10,j]
    # elements can be integers or LLVM ints
    #    with indexing being done either at compile time (Python int)
    #    or run time (LLVM int)
    last_elsl = None
    needstrided = False
    for i, val in enumerate(key):
        if isinteger(val):
            if last_elsl is None:
                last_elsl = i
        elif isinstance(val, (Ellipsis, slice)):
            if last_elsl is not None:
                needstrided = True
            needstrided = needstrided or _needstride(val)
        else:
            raise ValueError(_keymsg % val)

    # Return just the integers fields if needstrided not set
    if not needstrided:
        key = [_convert(x) for x in itertools.islice(key, lastint, None)]

    return needstrided, key


def _getitem_S(arr, key):
    return True, key


def from_Array(arr, key, char):
    if isinteger(key):
        return eval('from_%s_int' % char)(arr, key)
    elif isinstance(key, slice):
        if key == slice(None):
            return arr
        else:
            start, stop, step = adjust_slice(arr, key)
            if step == 1:
                return eval('from_%s_slice' % char)(arr, start, stop)
            else:
                return eval('Sarr_from_%s_slice' % char)(arr, start, stop, step)
    elif isiterable(key):
        # will be less than arr._nd or have '...' or ':'
        # at the end
        needstrided, key = eval("_getitem_%s" % char)(arr, key)
        if needstrided:
            return eval('Sarr_from_%s' % char)(arr, key)
        if len(key) > arr.nd:
            raise ValueError('Too many indicies')
        return eval('from_%s_ints' % char)(arr, key)
    else:
        raise ValueError(_keymsg % key)
