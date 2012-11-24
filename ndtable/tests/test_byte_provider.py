"""
Test the internals of constructing arrays from various byte
providers sources. These interfaces would not normally be exposed
to the end-user.
"""

import numpy as np
import ndtable.carray as ca

from ndtable.datashape import dshape
from ndtable.table import NDTable, NDArray
from ndtable.sources.canonical import PythonSource, ByteSource,\
CArraySource, ArraySource

def test_from_carray():
    c1 = ca.carray([1,2], ca.cparams(clevel=0, shuffle=False))
    b1 = CArraySource(c1)

    shape = dshape('2, int')
    NDArray.from_providers(shape, b1)

def test_from_numpy():
    c1 = np.array([1,2])
    b1 = ArraySource(c1)

    shape = dshape('2, int')
    NDArray.from_providers(shape, b1)

def test_from_bytes():
    #bits = bytes(np.ones((2,2), dtype=np.dtype('int32')).data)
    bits = b'\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00'
    b1 = ByteSource(bits)

    shape = dshape('2, 2, int')
    nd = NDArray.from_providers(shape, b1)

def test_from_python():
    b1 = PythonSource([1,2])
    b2 = PythonSource([3,4])

    shape = dshape('4, 2, int')
    NDArray.from_providers(shape, b1, b2)
