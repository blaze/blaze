"""
Blaze traits constituting sets of types.
"""

from itertools import chain

from .coretypes import *

__all__ = ['TypeSet', 'matches_typeset', 'signed', 'unsigned', 'integral',
           'floating', 'complexes', 'boolean', 'numeric', 'scalar']


class TypeSet(Mono):

    def __init__(self, *args):
        self._order = args
        self._set = set(args)

    @property
    def parameters(self):
        return tuple(self._order)

    def __contains__(self, val):
        return val in self._set

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self._set)

    def __or__(self, other):
        return TypeSet(*chain(self, other))

    def __iter__(self):
        return iter(self._order)

    def __len__(self):
        return len(self._set)


def matches_typeset(types, signature):
    """Match argument types to the parameter types of a signature"""
    match = True
    for a, b in zip(types, signature):
        check = isinstance(b, TypeSet)
        if check and (a not in b) or (not check and a != b):
            match = False
            break
    return match

#------------------------------------------------------------------------
# Type Sets
#------------------------------------------------------------------------

signed = TypeSet(int8, int16, int32, int64)
unsigned = TypeSet(uint8, uint16, uint32, uint64)
integral = TypeSet(*[x for t in zip(signed, unsigned) for x in t])
floating = TypeSet(float32, float64)
complexes = TypeSet(complex64, complex128)
boolean = TypeSet(bool_)
numeric = integral | floating | complexes
scalar = boolean | numeric