"""
Lift ckernels to their appropriate rank so they always consume the full array
arguments.
"""

from __future__ import absolute_import, division, print_function

from dynd import nd, ndt, _lowlevel

from ..traversal import visit


class CKernelLifter(object):
    """
    Lift ckernels to their appropriate rank so they always consume the
    full array arguments.

    If the environment defines 'stream-outer' as True, then the
    outermost dimension is skipped, so that the operation can be
    chunked along that dimension.
    """
    def __init__(self, env):
        self.env = env

    def get_arg_type(self, arg):
        dynd_types = self.env['dynd-types']
        if arg in dynd_types:
            return dynd_types[arg]
        else:
            return ndt.type(str(arg.type))

    def op_ckernel(self, op):
        op_ndim = len(op.type.shape)
        result_ndim = self.env.get('result-ndim', 0)
        ckernel, args = op.args
        in_types = [self.get_arg_type(arg) for arg in args[1:]]
        out_type = ndt.type(str(args[0].type))

        if isinstance(ckernel, dict):
            tag = ckernel['tag']
            if tag == 'elwise':
                ck = ckernel['ckernel']
                if op.metadata['rank'] < op_ndim and \
                        self.env.get('stream-outer', False) and result_ndim == op_ndim:
                    # Replace the leading dimension type with 'strided' in each operand
                    # if we're streaming it for processing BLZ
                    # TODO: Add dynd tp.subarray(N) function like datashape has
                    for i, tp in enumerate(in_types):
                        if tp.ndim == result_ndim:
                            in_types[i] = ndt.make_strided_dim(tp.element_type)
                    out_type = ndt.make_strided_dim(out_type.element_type)

                op.args[0] = _lowlevel.lift_arrfunc(ck, [out_type] + in_types)
            elif tag == 'reduction':
                ck = ckernel['ckernel']
                assoc = ckernel['assoc']
                comm = ckernel['comm']
                ident = ckernel['ident']
                ident = None if ident is None else nd.asarray(ident)
                axis = ckernel['axis']
                keepdims = ckernel['keepdims']
                op.args[0] = _lowlevel.lift_reduction_arrfunc(
                                ck, in_types[0],
                                axis=axis, keepdims=keepdims,
                                associative=assoc, commutative=comm,
                                reduction_identity=ident)
            elif tag == 'rolling':
                ck = ckernel['ckernel']
                window = ckernel['window']
                minp = ckernel['minp']
                if minp != 0:
                    raise ValueError('rolling window with minp != 0 not supported yet')
                op.args[0] = _lowlevel.make_rolling_arrfunc(out_type,
                                                                     in_types[0],
                                                                     ck, window)
            elif tag == 'ckfactory':
                ckfactory = ckernel['ckernel_factory']
                ck = ckfactory(out_type, *in_types)
                op.args[0] = ck
            else:
                raise RuntimeError('unnrecognized ckernel tag %s' % tag)
        else:
            op.args[0] = ckernel


def run(func, env):
    visit(CKernelLifter(env), func)
