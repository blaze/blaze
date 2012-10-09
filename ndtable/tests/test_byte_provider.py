import numpy as np
from ndtable.datashape import datashape
from ndtable.table import NDTable
from ndtable.datashape.unification import CannotEmbed
from ndtable.sources.canonical import PythonSource

from nose.tools import assert_raises

def setUp():
    na = np.ones((5,5), dtype=np.dtype('int32'))
    na.tofile('foo.npy')

def test_from_views():
    # Heterogenous Python lists of objects
    a = [1,2,3,4]
    b = [5,6,7,8]

    ai = PythonSource(a)
    bi = PythonSource(b)

    shape = datashape('8, object')
    table = NDTable.from_providers(shape, ai, bi)


def test_from_views_complex_dims():
    a = [1,2,3,4]
    b = [5,6,7,8]

    ai = PythonSource(a)
    bi = PythonSource(b)

    shape = datashape('2, Var(10), int32')
    table = NDTable.from_providers(shape, ai, bi)


def test_from_views_complex_dims():
    a = [1,2,3,4]
    b = [5,6,7,8]

    ai = PythonSource(a)
    bi = PythonSource(b)

    shape = datashape('2, Var(5), int32')
    table = NDTable.from_providers(shape, ai, bi)

def test_ragged():
    a = [1,2,3,4]
    b = [5,6]
    c = [7]

    ai = PythonSource(a)
    bi = PythonSource(b)
    ci = PythonSource(c)

    shape = datashape('3, Var(5), int32')
    table = NDTable.from_providers(shape, ai, bi, ci)

    assert not table.space.regular

def test_mismatch_inner():
    a = [1,2,3,4]
    b = [5,6]
    c = [7]

    ai = PythonSource(a)
    bi = PythonSource(b)
    ci = PythonSource(c)

    shape = datashape('3, Var(1), int32')
    with assert_raises(CannotEmbed):
        table = NDTable.from_providers(shape, ai, bi, ci)

def test_mismatch_outer():
    a = [1,2,3,4]
    b = [5,6]
    c = [7]

    ai = PythonSource(a)
    bi = PythonSource(b)
    ci = PythonSource(c)

    shape = datashape('2, Var(5), int32')
    with assert_raises(CannotEmbed):
        table = NDTable.from_providers(shape, ai, bi, ci)
