from __future__ import absolute_import, division, print_function

from blaze import Array, array
from dynd import nd
from datashape.dispatch import dispatch
from .data_descriptor import DDesc
from .as_py import ddesc_as_py


@dispatch(list, DDesc)
def like(a, b):
    return ddesc_as_py(b)

@dispatch(nd.array, DDesc)
def like(a, b):
    return b.dynd_arr()

@dispatch(Array, DDesc)
def like(a, b):
    return array(b)
