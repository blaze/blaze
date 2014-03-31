"""
Lift ckernels to their appropriate rank so they always consume the full array
arguments.
"""

from __future__ import absolute_import, division, print_function
from ....datadescriptor import DyND_DDesc, BLZ_DDesc

from dynd import nd

def prepare_local_execution(func, env):
    """
    Prepare for local execution
    """
    storage = env['storage']

    argdict = env['runtime.args']
    args = [argdict[arg] for arg in func.args]

    # If it's a BLZ output, we want an interpreter that streams
    # the processing through in chunks
    if storage is not None:
        if len(func.type.restype.shape) == 0:
            raise TypeError('Require an array, not a scalar, for outputting to BLZ')

        result_ndim = len(func.type.restype.shape)
        env['stream-outer'] = True
        env['result-ndim'] = result_ndim
    else:
        # Convert any persistent inputs to memory
        # TODO: should stream the computation in this case
        for i, arg in enumerate(args):
            if isinstance(arg._data, BLZ_DDesc):
                args[i] = arg[:]

    # Update environment with dynd type information
    dynd_types = dict((arg, get_dynd_type(array))
                          for arg, array in zip(func.args, args)
                              if isinstance(array._data, DyND_DDesc))
    env['dynd-types'] = dynd_types
    env['runtime.arglist'] = args


def get_dynd_type(array):
    return nd.type_of(array._data.dynd_arr())
