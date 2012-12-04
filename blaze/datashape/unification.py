"""
Unification is a generalization of Numpy broadcasting.

In Numpy we two arrays and broadcast them to yield similar
shaped arrays.

In Blaze we take two arrays with more complex datashapes and
unify the types prescribed by more complicated pattern matching
on the types.

"""

from numpy import promote_types
from coretypes import Fixed, Range, TypeVar, Record, \
    CType, Enum, top, dynamic

class Incommensurable(Exception):
    def __init__(self, space, dim):
        self.space = space
        self.dim   = dim

    def __str__(self):
        return "No way of unifying (%s) (%s)" % (
            self.space, self.dim
        )

def unify(a, b):
    """
    Unification of Datashapes.
    """
    ta = type(a)
    tb = type(b)

    # --

    # Unification over BlazeT has two zeros

    if ta == top or tb == top:
        return top

    if ta == dynamic or tb == dynamic:
        return top

    # --

    if (ta,tb) == (Fixed, Fixed):
        if a.val == b.val:
            return Fixed(a.val)
        else:
            return Enum(a.val, b.val)

    # --

    if (ta,tb) == (TypeVar, Fixed):
        return TypeVar('x0')

    if (ta,tb) == (Fixed, TypeVar):
        return TypeVar('x0')

    # --

    if (ta,tb) == (Record, Record):
        c = a.d.items() + b.d.items()
        return Record(**dict(c))

    # --

    if (ta,tb) == (Fixed, Range):
        return Range(min(a.val, b.lower), max(a.val, b.upper))

    if (ta,tb) == (Range, Fixed):
        return Range(min(a.lower, b.val), max(a.val, b.val))

    if (ta,tb) == (Range, Range):
        return Range(min(a.lower, b.lower), max(b.upper, b.upper))

    # --

    #if (ta,tb) == (Union, Union):
        #return Union(a.parameters + b.parameters)

    # --

    if (ta,tb) == (CType, CType):
        return CType.from_str(promote_types(a.name, b.name).name)

    raise Incommensurable(a,b)
