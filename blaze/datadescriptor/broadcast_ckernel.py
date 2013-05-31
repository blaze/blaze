from __future__ import absolute_import

__all__ = ['execute_unary_single']

from ..error import BroadcastError

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
                se = src.element_reader(0)
                src_ptr = se.read_single(())
                with dst.element_write_iter() as de:
                    for dst_ptr in de:
                        ck(dst_ptr, src_ptr)
            else:
                se = src.element_read_iter()
                with dst.element_write_iter() as de:
                    for dst_ptr in de, src_ptr in se:
                        ck(dst_ptr, src_ptr)
        else:
            # Use the Python-level looping constructs
            # for processing higher numbers of dimensions.
            if src_dim_size == 1:
                src_dd = src[0]
                for dst_dd in dst:
                    execute_unary_single(dst_dd, src_dd, dst_ds, src_ds, ck)
            else:
                for dst_dd, src_dd in src, dst:
                    execute_unary_single(dst_dd, src_dd, dst_ds, src_ds, ck)
