from __future__ import absolute_import, division, print_function

from dynd import nd, ndt

from .data_descriptor import I_DDesc
from .blz_data_descriptor import BLZ_DDesc


def ddesc_as_py(ddesc):
    """
    Converts the data in a data descriptor into Python
    types. This uses the data_descriptor iteration methods,
    so is not expected to be fast. Its main initial purpose
    is to assist with writing unit tests.
    """
    # TODO: This function should probably be removed.
    if not isinstance(ddesc, I_DDesc):
        raise TypeError('expected I_DDesc instance, got %r' % type(ddesc))

    if isinstance(ddesc, BLZ_DDesc):
        return [ddesc_as_py(child_ddesc) for child_ddesc in ddesc]

    if ddesc.capabilities.deferred:
        from blaze import Array, eval
        ddesc = eval(Array(ddesc)).ddesc
    return nd.as_py(ddesc.dynd_arr())
