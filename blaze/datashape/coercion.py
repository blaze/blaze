# -*- coding: utf-8 -*-

"""
This module implements type coercion rules for data shapes.

Note that transitive coercions could be supported, but we decide not to since
it may involve calling a whole bunch of functions with a whole bunch of types
to figure out whether this is possible in the face of polymorphic overloads.
"""

from itertools import chain

from blaze import error
from blaze.overloading import overload
from . import coretypes as T
from .traits import *

#------------------------------------------------------------------------
# Overloadable coercion function
#------------------------------------------------------------------------

@overload('Type A -> Type B -> int')
def coerce(a, b):
    """
    Determine a coercion cost for coercing type `a` to type `b`
    """
    raise error.CoercionError(a, b)

#------------------------------------------------------------------------
# Default coercion rules
#------------------------------------------------------------------------

coercions = set(map(tuple, ['ui', 'if', 'uf', 'ic', 'uc', 'fc']))
order = list(chain(integral, floating, complexes))

numeric_typecodes = {
    signed: 'i',
    unsigned: 'u',
    floating: 'f',
    complexes: 'c',
}
numeric_typecode = numeric_typecodes.__getitem__

@overload('Type (A : numeric) -> Type (B : numeric) -> int')
def coerce(a, b):
    if a == b:
        return 0

    code1, code2 = numeric_typecode(a), numeric_typecode(b)
    if b.itemsize < a.itemsize or (code1, code2) not in coercions:
        raise error.CoercionError(a, b)
    else:
        return order.index(b) - order.index(a)

@overload('Type (A : numeric) -> Type (B : object) -> int')
def coerce(a, b):
    dist = len(order) - order.index(a)
    return dist + 100 # this is somewhat arbitrary

#---------
#  Boolean

@overload('Type (A : boolean) -> Type (B : signed) -> int')
def coerce(a, b):
    return list(signed).index(b)

@overload('Type (A : boolean) -> Type (B : object) -> int')
def coerce(a, b):
    return coerce(a, T.int8) + coerce(T.int8, b)