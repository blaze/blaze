import numpy as np
from ndtable.datashape import datashape
from ndtable.table import NDTable
from ndtable.sources.canonical import PythonSource, ByteSource

from nose.tools import assert_raises

def test_from_bytes():
    #bits = np.ones((2,2), dtype=np.dtype('int32')).data
    bits = bytes('\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00')

    b1 = ByteSource(bits)
    b2 = ByteSource(bits)

    # concat row-wise
    shape = datashape('4, 2, int32')
    NDTable.from_providers(shape, b1, b2)
