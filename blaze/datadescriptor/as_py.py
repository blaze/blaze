from __future__ import absolute_import, division, print_function

from dynd import nd, ndt

from .data_descriptor import I_DDesc
from .blz_data_descriptor import BLZ_DDesc


def dd_as_py(dd):
    """
    Converts the data in a data descriptor into Python
    types. This uses the data_descriptor iteration methods,
    so is not expected to be fast. Its main initial purpose
    is to assist with writing unit tests.
    """
    # TODO: This function should probably be removed.
    if not isinstance(dd, I_DDesc):
        raise TypeError('expected I_DDesc instance, got %r' % type(dd))

    if isinstance(dd, BLZ_DDesc):
        return [dd_as_py(child_dd) for child_dd in dd]

    if dd.capabilities.deferred:
        from blaze import Array, eval
        dd = eval(Array(dd))._data
    return nd.as_py(dd.dynd_arr())
