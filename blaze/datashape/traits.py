"""
Blaze traits constituting sets of types.
"""

from itertools import chain

from blaze import error
from .coretypes import *

__all__ = ['TypeSet', 'matches_typeset', 'signed', 'unsigned', 'integral',
           'floating', 'complexes', 'boolean', 'numeric', 'scalar']


class TypeSet(Mono):
    """
    Create a new set of types. Keyword argument 'name' may create a registered
    typeset for use in datashape type strings.
    """

    def __init__(self, *args, **kwds):
        self._order = args
        self._set = set(args)
        self.name = kwds.get('name')
        if self.name:
            register_typeset(self.name, self)

    @property
    def types(self):
        return self._order

    @property
    def parameters(self):
        return tuple(self._order)

    def __eq__(self, other):
        return (isinstance(other, type(self)) and
                self.parameters == other.parameters)

    def __contains__(self, val):
        return val in self._set

    def __repr__(self):
        if self.name:
            return '{%s}' % (self.name,)
        return "%s(%s, name=%s)" % (self.__class__.__name__, self._set, self.name)

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

class TypesetRegistry(object):
    def __init__(self):
        self.registry = {}
        self.lookup = self.registry.get

    def register_typeset(self, name, typeset):
        if name in typeset:
            raise error.BlazeTypeError("TypeSet %s already defined" % name)
        self.registry[name] = typeset
        return typeset

    def __getitem__(self, key):
        value = self.lookup(key)
        if value is None:
            raise KeyError(key)
        return value

registry = TypesetRegistry()
register_typeset = registry.register_typeset
lookup = registry.lookup

#------------------------------------------------------------------------
# Default Type Sets
#------------------------------------------------------------------------

signed = TypeSet(int8, int16, int32, int64, name='signed')
unsigned = TypeSet(uint8, uint16, uint32, uint64, name='unsigned')
integral = TypeSet(*[x for t in zip(signed, unsigned) for x in t],
                   name='integral')
floating = TypeSet(float32, float64, name='floating')
complexes = TypeSet(complex64, complex128, name='complexes')
boolean = TypeSet(bool_, name='boolean')
numeric = TypeSet(*integral | floating | complexes, name='numeric')
scalar = TypeSet(*boolean | numeric, name='signed')