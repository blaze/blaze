from .llvm_array import (array_type, const_intp, auto_const_intp, 
                         store_at, load_at, get_shape_ptr,
                         get_strides_ptr, sizeof)
from llvm.core import Constant, Type
import llvm.core as lc

msg = "Unsupported getitem value %s"

# produce strided array view from contiguous

def adjust_slice(key, nd):
    start = key.start
    if start is None:
        start = 0
    while start < 0:
        start += nd

    end = key.end
    if end is None:
        end = nd

    while end < 0:
        end += nd

    step = key.step
    if step is None:
        step = 1

    return start, end, step


def Sarr_from_C(arr, key):
    raise NotImplementedError


def Sarr_from_C_slice(arr, start, stop, step):
    raise NotImplementedError


def from_C_int(arr, index):
    return from_C_ints(arr, (index,))

def from_C_ints(arr, key):
    builder = arr.builder
    num = len(key)
    newnd = arr.nd - num
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

def from_C(arr, key):
    if hasattr(key, '__index__'):
        key = key.__index()
        return from_C_int(arr, key)
    elif isinstance(key, slice):
        if key == slice(None):
            return arr
        else:
            start, stop, step = adjust_slice(key)
            if step == 1:
                return from_C_slice(arr, start, end)
            else:
                return Sarr_from_C_slice(arr, start, end, step)
    elif isinstance(key, tuple):
        # will be less than arr._nd or have '...' or ':'
        # at the end
        lastint = None
        needstrided = False
        for i, val in enumerate(key):
            if hasattr(val, '__index__'):
                if lastint is not None:
                    needstrided = True
            elif isinstance(val, (Ellipsis, slice)):
                if lastint is None:
                    lastint = i
            else:
                raise ValueError(msg % val)
        if needstrided:
            return Sarr_from_C(arr, key)
        # get just the integers
        key = [x.__index__() for x in key[slice(None, lastint)]]
        if len(key) > arr.nd:
            raise ValueError('Too many indicies')
        return from_C_ints(arr, key)
    else:
        raise ValueError(msg % key)


def from_F(arr, key):
    raise NotImplementedError

def from_S(arr, key):
    raise NotImplementedError