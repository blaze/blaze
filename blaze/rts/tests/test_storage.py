from ctypes import c_char
from blaze.rts.storage import Heap, Arena, address_of_buffer,\
    allocate_numpy, allocate_raw, finalize

import numpy as np

from StringIO import StringIO
from string import letters

from unittest import skip, skipIf

try:
    from numba.decorators import autojit
    have_numba = True
except ImportError:
    have_numba = False

try:
    import iopro
    have_iopro = True
except ImportError:
    have_iopro = False

def test_address_mmap():
    # never actually do this
    s = 'foo'

    a = Arena(3)
    a.block[0:3] = s

    ptr, size = address_of_buffer(a.block)
    assert (c_char*3).from_address(ptr).value == s

def test_malloc():
    h = Heap()
    _, block, _ = allocate_raw(h, 1000)

def test_free():
    h = Heap()

    addr1, block1, c1ptr = allocate_raw(h, 5)
    addr2, block2, c2ptr = allocate_raw(h, 5)

    assert len(h._arenas) == 1
    h.free(block2)

    addr3, block2, c2ptr = allocate_raw(h, 5)

    # Blocks get merged when free'd so that align blocks
    assert addr3 == addr2

#------------------------------------------------------------------------
# IOPro Prototype
#------------------------------------------------------------------------

@skipIf(not have_iopro, "iopro not installed")
def test_iopro():
    # this is kind of stupid right now because it does a copy,
    # but Tight IOPro integration will be a priority...

    h = Heap()

    s = StringIO(','.join(letters))

    data = iopro.genfromtxt(s, dtype='c', delimiter=",")

    addr, block = allocate_numpy(h, data.dtype, data.shape)
    block[:] = data[:]

    assert not block.flags['OWNDATA']
    assert block.ctypes.data == addr

    assert len(h._arenas) == 1
    assert block.nbytes < h._lengths[0]

    finalize(h)

#------------------------------------------------------------------------
# Numba Prototype
#------------------------------------------------------------------------

@skip
@skipIf(not have_numba, "numba not installed")
def test_numba():

    @autojit
    def fill(x):
        for i in range(25):
            x[i] = i

    h = Heap()
    addr, block = allocate_numpy(h, np.dtype('int'), (25, ))
    fill(block)

    finalize(h)
