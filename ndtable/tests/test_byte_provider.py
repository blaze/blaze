import numpy as np
from ndtable.datashape import datashape
from ndtable.table import NDTable
from ndtable.datashape.embedding import can_embed, CannotEmbed
from ndtable.sources.canonical import PythonSource, ByteSource

from nose.tools import assert_raises

def test_from_bytes():
    #bits = np.ones((2,2), dtype=np.dtype('int32')).data
    bits = bytes('\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00')

    b1 = ByteSource(bits)
    b2 = ByteSource(bits)

    # concat row-wise
    shape = datashape('4, 2, int32')

    # Can we devise a way of concatenating:
    #   (2,2,int32) + (2,2,int32) = (4,2,int32)
    #   (2,2,int32) + (2,2,int32) = (2,4,int32)
    #   (2,int32)   + (2,2,int32) = (3,2,int32)

    NDTable.from_providers(shape, b1, b2)

#def test_from_views():
    ## Heterogenous Python lists of objects
    #a = [1,2,3,4]
    #b = [5,6,7,8]

    #ai = PythonSource(a)
    #bi = PythonSource(b)

    #shape = datashape('8, object')
    #table = NDTable.from_providers(shape, ai, bi)


#def test_from_views_complex_dims():
    #a = [1,2,3,4]
    #b = [5,6,7,8]

    #ai = PythonSource(a)
    #bi = PythonSource(b)

    #shape = datashape('8, object')
    #table = NDTable.from_providers(shape, ai, bi)


#def test_from_views_complex_dims():
    #a = [1,2,3,4]
    #b = [5,6,7,8]

    #ai = PythonSource(a)
    #bi = PythonSource(b)

    #shape = datashape('8, object')
    #table = NDTable.from_providers(shape, ai, bi)

#def test_ragged():
    #a = [1,2,3,4]
    #b = [5,6]
    #c = [7]

    #ai = PythonSource(a)
    #bi = PythonSource(b)
    #ci = PythonSource(c)

    #shape = datashape('Var(7), object')
    #table = NDTable.from_providers(shape, ai, bi, ci)

    #assert not table.space.regular

#def test_mismatch_inner():
    #a = [1,2,3,4]
    #b = [5,6]
    #c = [7]

    #ai = PythonSource(a)
    #bi = PythonSource(b)
    #ci = PythonSource(c)

    #shape = datashape('3, Var(1), int32')
    #with assert_raises(CannotEmbed):
        #table = NDTable.from_providers(shape, ai, bi, ci)

#def test_mismatch_outer():
    #a = [1,2,3,4]
    #b = [5,6]
    #c = [7]

    #ai = PythonSource(a)
    #bi = PythonSource(b)
    #ci = PythonSource(c)

    #shape = datashape('2, Var(5), int32')
    #with assert_raises(CannotEmbed):
        #table = NDTable.from_providers(shape, ai, bi, ci)
