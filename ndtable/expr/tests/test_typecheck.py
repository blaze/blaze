"""
Tests for the core type checker, using Python's own system to
test the algorithm.
"""

from ndtable.datashape.unification import Incommensurable
from ndtable.expr.typechecker import tyeval, typesystem
from nose.tools import assert_raises

#------------------------------------------------------------------------
# Universe of Discourse
#------------------------------------------------------------------------

bools     = set([bool])
ints      = set([int])
floats    = set([float])
universal = set([object])

numerics = ints | floats

#------------------------------------------------------------------------
# Unification
#------------------------------------------------------------------------

class dynamic(object):
    def __repr__(self):
        return '?'

def unify(context, a,b):
    """
    Very simple unification.
    """

    if (a,b) == (int, int):
        return int

    if (a,b) == (int, float):
        return float

    if (a,b) == (float, int):
        return float

    raise Incommensurable(a,b)

#------------------------------------------------------------------------
# Type System
#------------------------------------------------------------------------

PythonT = typesystem(unifier=unify, top=object, dynamic=dynamic, typeof=type)

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

def test_simple_uni():
    res = tyeval('a -> a', [1], [ints], PythonT)
    assert res.dom == [int]
    assert res.cod == int

def test_simple_bi():
    res = tyeval('a -> a -> a', [1, 2], [ints, ints], PythonT)
    assert res.dom     == [int, int]
    assert res.cod     == int
    assert res.dynamic == False

def test_simple_bi_free():
    res = tyeval('a -> a -> b', [1, 2], [ints, ints], PythonT)
    assert res.dynamic == True

def test_simple_bi_domain():
    res = tyeval('a -> a -> a', [1, 2], [numerics, numerics], PythonT)
    assert res.dom     == [int, int]
    assert res.cod     == int
    assert res.dynamic == False

def test_simple_unsatisfiable():
    with assert_raises(TypeError):
        res = tyeval('a -> a -> b', [1, False], [ints, ints], PythonT)

def test_simple_unification():
    res = tyeval('a -> a -> a', [1, 3.14], [numerics, numerics], PythonT)
    assert res.dom     == [float, float]
    assert res.cod     == float
    assert res.dynamic == False

def test_complext_unification():
    res = tyeval('a -> b -> a -> b', [1, False, 2], \
            [numerics, bools, numerics, bools], PythonT)
    assert res.dom     == [int, bool, int]
    assert res.cod     == bool
    assert res.dynamic == False

# def test_commutativity():
#     res = tyeval('a -> b -> b', [True, 1], [numerics, bools], PythonT,
#             commutative=True)
#     assert res.dom     == [int, bool]
#     assert res.cod     == bool
#     assert res.dynamic == False
