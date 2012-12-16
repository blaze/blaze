"""
Tests for the core type checker, using Python's own system to
test the algorithm.
"""

from blaze.datashape.unification import Incommensurable
from blaze.expr.typechecker import infer, typesystem
from nose.tools import assert_raises

#------------------------------------------------------------------------
# Toy System
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

def unify(a,b):
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
    res = infer('a -> a', [1], {'a': ints}, PythonT)
