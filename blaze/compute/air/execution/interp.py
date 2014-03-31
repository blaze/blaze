"""CKernel evaluation of blaze AIR."""

from __future__ import absolute_import, division, print_function

import operator

from dynd import nd, ndt
import blaze
import blz
import datashape

from ..traversal import visit
from ....datadescriptor import DyND_DDesc, BLZ_DDesc


def interpret(func, env, ddesc=None, **kwds):
    args = env['runtime.arglist']

    if ddesc is None:
        # Evaluate once
        values = dict(zip(func.args, args))
        interp = CKernelInterp(values)
        visit(interp, func)
        return interp.result
    else:
        result_ndim = env['result-ndim']

        res_shape, res_dt = datashape.to_numpy(func.type.restype)
        dim_size = operator.index(res_shape[0])
        row_size = ndt.type(str(func.type.restype.subarray(1))).data_size
        chunk_size = min(max(1, (1024*1024) // row_size), dim_size)
        # Evaluate by streaming the outermost dimension,
        # and using the BLZ data descriptor's append
        ddesc.blzarr = blz.zeros((0,)+res_shape[1:], res_dt,
                           rootdir=ddesc.path)
        # Loop through all the chunks
        for chunk_start in range(0, dim_size, chunk_size):
            # Tell the interpreter which chunk size to use (last
            # chunk might be smaller)
            chunk_size = min(chunk_size, dim_size - chunk_start)
            # Evaluate the chunk
            args_chunk = [arg[chunk_start:chunk_start+chunk_size]
                            if len(arg.dshape.shape) == result_ndim
                            else arg for arg in args]
            values = dict(zip(func.args, args_chunk))
            interp = CKernelChunkInterp(values, chunk_size, result_ndim)
            visit(interp, func)
            chunk = interp.result.ddesc.dynd_arr()
            ddesc.append(chunk)

        return blaze.Array(ddesc)


class CKernelInterp(object):
    """
    Interpret low-level AIR in the most straightforward way possible.

    Low-level AIR contains the following operations:

        alloc/dealloc
        ckernel

    There is a huge number of things we can still do, like blocking and
    parallelism.

    Blocking
    ========
    This should probably happen through a "blocking-ckernel" wrapper

    Parallelism
    ===========
    Both data-parallelism by executing ckernels over slices, and executing
    disjoint sub-expressions in parallel.
    """

    def __init__(self, values):
        self.values = values # { Op : py_val }

    def op_alloc(self, op):
        dshape = op.type
        ddesc = op.metadata.get('ddesc') # TODO: ddesc!
        self.values[op] = blaze.empty(dshape, ddesc=ddesc)

    def op_dealloc(self, op):
        alloc, = op.args
        del self.values[alloc]

    def op_convert(self, op):
        input = self.values[op.args[0]]
        input = input.ddesc.dynd_arr()
        result = nd.array(input, type=ndt.type(str(op.type)))
        result = blaze.Array(DyND_DDesc(result))
        self.values[op] = result

    def op_pykernel(self, op):
        pykernel, opargs = op.args
        args = [self.values[arg] for arg in opargs]
        result = pykernel(*args)
        self.values[op] = result

    def op_kernel(self, op):
        raise RuntimeError("Shouldn't be seeing a kernel here...", op)

    def op_ckernel(self, op):
        raise RuntimeError("Shouldn't be seeing a ckernel here...", op)

    def op_ret(self, op):
        retvar = op.args[0]
        self.result = self.values[retvar]


class CKernelChunkInterp(CKernelInterp):
    """
    Like CKernelInterp, but for processing one chunk.
    """

    def __init__(self, values, chunk_size, result_ndim):
        self.values = values # { Op : py_val }
        self.chunk_size = chunk_size
        self.result_ndim = result_ndim

    def op_alloc(self, op):
        dshape = op.type
        # Allocate a chunk instead of the whole thing
        if len(dshape.shape) == self.result_ndim:
            chunk = nd.empty(self.chunk_size, str(dshape.subarray(1)))
        else:
            chunk = nd.empty(str(dshape))
        self.values[op] = blaze.array(chunk)
