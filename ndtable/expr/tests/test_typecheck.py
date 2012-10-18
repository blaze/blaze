"""
Tests for the core type checker, using Python's own system to
test the algorithm.
"""

from ndtable.datashape.unification import Incommensurable
from ndtable.expr.typechecker import typecheck, typesystem

# Use the Python types for testing.
bools     = set([bool])
ints      = set([int])
floats    = set([float])
universal = set([object])

numerics = ints | floats
typeof = type

def unify(a,b):

    if (a,b) == (int, int):
        return int

    if (a,b) == (int, float):
        return float

    if (a,b) == (float, int):
        return float

    raise Incommensurable(a,b)

PythonT = typesystem(unify, object, typeof)

def test_simple_uni():
    res = typecheck('a -> a', [1], [ints], PythonT)
    assert res.dom == [int]
    assert res.cod == int

def test_simple_bi():
    res = typecheck('a -> a -> a', [1, 2], [ints, ints], PythonT)
    assert res.dom    == [int, int]
    assert res.cod    == int
    assert res.opaque == False

def test_simple_bi_free():
    res = typecheck('a -> a -> b', [1, 2], [ints, ints], PythonT)
    assert res.opaque == True

def test_simple_unification():
    res = typecheck('a -> a -> a', [1, 3.14], [numerics, numerics], PythonT)
    assert res.dom    == [float, float]
    assert res.cod    == float
    assert res.opaque == False
