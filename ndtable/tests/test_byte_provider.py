"""
Test the internals of constructing arrays from various byte
providers sources. These interfaces would not normally be exposed
to the end-user.
"""

import numpy as np
import carray as ca

from ndtable.datashape import datashape
from ndtable.table import NDTable, NDArray
from ndtable.sources.canonical import PythonSource, ByteSource,\
    CArraySource, ArraySource

def test_from_carray():

    # the simplest Carray instance
    c1 = ca.carray([1,2], ca.cparams(clevel=0, shuffle=False))
    b1 = CArraySource(c1)

    # concat row-wise
    shape = datashape('2, int64')
    NDArray.from_providers(shape, b1)

def test_from_numpy():

    c1 = np.array([1,2])
    b1 = ArraySource(c1)

    # concat row-wise
    shape = datashape('2, int64')
    NDArray.from_providers(shape, b1)

def test_from_bytes():

    #bits = bytes(np.ones((2,2), dtype=np.dtype('int32')).data)
    bits = bytes('\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00')

    b1 = ByteSource(bits)
    b2 = ByteSource(bits)

    # concat row-wise
    shape = datashape('4, 2, int32')
    NDArray.from_providers(shape, b1, b2)


def test_from_python():

    b1 = PythonSource([1,2])
    b2 = PythonSource([3,4])

    # concat row-wise
    shape = datashape('4, 2, int32')
    NDArray.from_providers(shape, b1, b2)
