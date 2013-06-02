from __future__ import absolute_import

__all__ = ['execute_unary_single', 'execute_expr_single']

import ctypes

from ..error import BroadcastError
from ..py3help import izip

from .as_py import dd_as_py

def execute_unary_single(dst, src, dst_ds, src_ds, ck):
    """Executes a unary single kernel on all elements of
    the src data descriptor, writing to the dst data
    descriptor.

    Parameters
    ----------
    dst : data descriptor
        A writable data descriptor for output.
    src : data descriptor
        A readable data descriptor for input.
    dst_ds : data shape
        The data shape the ckernel writes to.
        This must be a suffix of dst's data shape.
    src_ds : data shape
        The data shape the ckernel reads from.
        This must be a suffix of src's data shape.
    ck : ckernel
        The kernel object to execute, with a
        UnarySingleOperation function prototype
    """
    src_ndim = len(src.dshape) - len(src_ds)
    dst_ndim = len(dst.dshape) - len(dst_ds)
    if src.dshape.subarray(src_ndim) != src_ds:
        raise TypeError(('Kernel dshape %s must be a suffix ' +
                        'of data descriptor dshape %s') % (src_ds, src.dshape))
    if dst.dshape.subarray(dst_ndim) != dst_ds:
        raise TypeError(('Kernel dshape %s must be a suffix ' +
                        'of data descriptor dshape %s') % (dst_ds, dst.dshape))

    if src_ndim < dst_ndim:
        # Broadcast the src data descriptor across
        # the outermost dst dimension
        if dst_ndim == 1:
            # If there's a one-dimensional loop left,
            # use the element write iter to process
            # it.
            se = src.element_reader(0)
            src_ptr = se.read_single(())
            with dst.element_write_iter() as de:
                for dst_ptr in de:
                    ck(dst_ptr, src_ptr)
        else:
            # Use the Python-level looping constructs
            # for processing higher numbers of dimensions.
            for dd in dst:
                execute_unary_single(dd, src, dst_ds, src_ds, ck)
    elif src_ndim > dst_ndim:
        raise BroadcastError('Cannot broadcast into a dshape with fewer dimensions')
    elif dst_ndim == 0:
        # Call the kernel once
        se = src.element_reader(0)
        src_ptr = se.read_single(())
        de = dst.element_writer(0)
        with de.buffered_ptr(()) as dst_ptr:
            ck(dst_ptr, src_ptr)
    else:
        dst_dim_size, src_dim_size = len(dst), len(src)
        if src_dim_size not in [1, dst_dim_size]:
            raise BroadcastError(('Cannot broadcast dimension of ' +
                        'size %d into size %d') % (src_dim_size, dst_dim_size))
        # Broadcast the outermost dimension of src
        # to the outermost dimension of dst
        if dst_ndim == 1:
            # Use the element pointer interfaces for the last dimension
            if src_dim_size == 1:
                se = src.element_reader(1)
                src_ptr = se.read_single((0,))
                with dst.element_write_iter() as de:
                    for dst_ptr in de:
                        ck(dst_ptr, src_ptr)
            else:
                se = src.element_read_iter()
                with dst.element_write_iter() as de:
                    for dst_ptr, src_ptr in izip(de, se):
                        ck(dst_ptr, src_ptr)
        else:
            # Use the Python-level looping constructs
            # for processing higher numbers of dimensions.
            if src_dim_size == 1:
                src_dd = src[0]
                for dst_dd in dst:
                    execute_unary_single(dst_dd, src_dd, dst_ds, src_ds, ck)
            else:
                for dst_dd, src_dd in izip(dst, src):
                    execute_unary_single(dst_dd, src_dd, dst_ds, src_ds, ck)

def execute_expr_single(dst, src_list, dst_ds, src_ds_list, ck):
    """Executes an expr single kernel on all elements of
    the src data descriptors, writing to the dst data
    descriptor. The number of src data descriptors and
    data shapes must match that required by the ckernel.

    Parameters
    ----------
    dst : data descriptor
        A writable data descriptor for output.
    src_list : list of data descriptor
        A list of readable data descriptors for input.
    dst_ds : data shape
        The data shape the ckernel writes to.
        This must be a suffix of dst's data shape.
    src_ds_list : list of data shape
        The data shape the ckernel reads from.
        Each must be a suffix of corresponding
        src's data shape.
    ck : ckernel
        The kernel object to execute, with a
        ExprSingleOperation function prototype
    """
    if len(src_list) != len(src_ds_list):
        raise ValueError('The length of src_list and src_ds_list must match')
    src_ndim_list = [len(src.dshape) - len(src_ds)
                    for src, src_ds in izip(src_list, src_ds_list)]
    dst_ndim = len(dst.dshape) - len(dst_ds)
    for src, src_ds, src_ndim in izip(src_list, src_ds_list, src_ndim_list):
        if src.dshape.subarray(src_ndim) != src_ds:
            raise TypeError(('Kernel dshape %s must be a suffix ' +
                            'of data descriptor dshape %s') % (src_ds, src.dshape))
    if dst.dshape.subarray(dst_ndim) != dst_ds:
        raise TypeError(('Kernel dshape %s must be a suffix ' +
                        'of data descriptor dshape %s') % (dst_ds, dst.dshape))

    SINGLE = 1
    ITER = 2
    src_work_list = []
    # Process all the src argument iterators/elements
    for src, src_ndim in izip(src_list, src_ndim_list):
        if src_ndim < dst_ndim:
            # Broadcast the src data descriptor across
            # the outermost dst dimension
            if dst_ndim == 1:
                # If there's a one-dimensional loop left,
                # use the element interfaces to process it
                se = src.element_reader(0)
                src_ptr = se.read_single(())
                src_work_list.append((SINGLE, src_ptr, se))
            else:
                src_work_list.append((SINGLE, src, None))
        elif src_ndim > dst_ndim:
            raise BroadcastError('Cannot broadcast into a dshape with fewer dimensions')
        elif dst_ndim == 0:
            # Call the kernel once
            se = src.element_reader(0)
            src_ptr = se.read_single(())
            src_work_list.append((SINGLE, src_ptr, se))
        else:
            dst_dim_size, src_dim_size = len(dst), len(src)
            if src_dim_size not in [1, dst_dim_size]:
                raise BroadcastError(('Cannot broadcast dimension of ' +
                            'size %d into size %d') % (src_dim_size, dst_dim_size))
            # Broadcast the outermost dimension of src
            # to the outermost dimension of dst
            if dst_ndim == 1:
                # Use the element pointer interfaces for the last dimension
                if src_dim_size == 1:
                    se = src.element_reader(1)
                    src_ptr = se.read_single((0,))
                    src_work_list.append((SINGLE, src_ptr, se))
                else:
                    se = src.element_read_iter()
                    src_work_list.append((ITER, se, None))
            else:
                # Use the Python-level looping constructs
                # for processing higher numbers of dimensions.
                if src_dim_size == 1:
                    src_work_list.append((SINGLE, src[0], None))
                else:
                    src_work_list.append((ITER, src.__iter__(), None))
    # Loop through the outermost dimension
    # Broadcast the src data descriptor across
    # the outermost dst dimension
    if dst_ndim == 0:
        src_ptr_arr = (ctypes.c_void_p * len(src_work_list))()
        for i, (tp, obj, aux) in enumerate(src_work_list):
            src_ptr_arr[i] = ctypes.c_void_p(obj)
        de = dst.element_writer(0)
        with de.buffered_ptr(()) as dst_ptr:
            ck(dst_ptr, ctypes.cast(src_ptr_arr, ctypes.POINTER(ctypes.c_void_p)))
    elif dst_ndim == 1:
        # If there's a one-dimensional loop left,
        # use the element write iter to process
        # it.
        src_ptr_arr = (ctypes.c_void_p * len(src_work_list))()
        with dst.element_write_iter() as de:
            for dst_ptr in de:
                for i, (tp, obj, aux) in enumerate(src_work_list):
                    if tp == SINGLE:
                        src_ptr_arr[i] = ctypes.c_void_p(obj)
                    else:
                        src_ptr_arr[i] = ctypes.c_void_p(next(obj))
                    #print('src ptr', i, ':', hex(src_ptr_arr[i]))
                #print('dst ptr', hex(dst_ptr))
                ck(dst_ptr, src_ptr_arr)
    else:
        # Use the Python-level looping constructs
        # for processing higher numbers of dimensions.
        for dd in dst:
            src_dd_list = []
            for tp, obj, aux in src_work_list:
                if tp == SINGLE:
                    src_dd_list.append(obj)
                else:
                    src_dd_list.append(next(obj))
            execute_expr_single(dd, src_dd_list, dst_ds, src_ds_list, ck)
