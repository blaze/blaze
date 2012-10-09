"""
Unification is a generalization of Numpy broadcasting.

In Numpy we two arrays and broadcast them to yield similar shaped
object::

    A      (2d array):  5 x 4
    B      (1d array):      1
    Result (2d array):  5 x 4

In Blaze we take two arrays with more complex datashapes and
merge them for::

    A                :  3, 2, Var(3)
    B                :     1, Var(4)
    Result           :  3, 2, Var(4)

"""

from numpy import promote_types
from coretypes import Fixed, Var, TypeVar, Record, \
    DataShape, CType

class CannotEmbed(Exception):
    def __init__(self, space, dim):
        self.space = space
        self.dim   = dim

    def __str__(self):
        return "Cannot embed space of values (%r) in (%r)" % (
            self.space, self.dim
        )

class Incommensurable(Exception):
    def __init__(self, space, dim):
        self.space = space
        self.dim   = dim

    def __str__(self):
        return "No way of unifying (%s) (%s)" % (
            self.space, self.dim
        )

def describe(obj):
    # circular references...
    from ndtable.table import NDTable

    if isinstance(obj, DataShape):
        return obj

    elif isinstance(obj, list):
        return Fixed(len(obj))

    elif isinstance(obj, tuple):
        return Fixed(len(obj))

    elif isinstance(obj, NDTable):
        return obj.datashape

def can_embed(obj, dim2):
    """
    Can we embed a ``obj`` inside of the space specified by the outer
    dimension ``dim``.
    """
    dim1 = describe(obj)

    # We want explicit fallthrough
    if isinstance(dim1, Fixed):

        if isinstance(dim2, Fixed):
            if dim1 == dim2:
                return True
            else:
                return False

        if isinstance(dim2, Var):
            if dim2.lower < dim1.val < dim2.upper:
                return True
            else:
                return False

        if isinstance(dim2, TypeVar):
            return True

    if isinstance(dim1, Record):

        if isinstance(dim2, Record):
            # is superset
            return set(dim1.k) >= set(dim2.k)

    if isinstance(dim1, TypeVar):
        return True

    raise CannotEmbed(dim1, dim2)

# I miss Haskell pattern matching. :`(
def union(dim1, dim2):
    x  , y  = dim1[0]  , dim2[0]
    xs , ys = dim1[1:] , dim2[1:]

    z = unify(x,y)

    if xs and ys:
        return union(xs, ys)*z
    else:
        return z

def unify(a,b):
    """
    Defines the unification of datashapes.
    """
    ta = type(a)
    tb = type(b)

    if (ta,tb) == (Fixed, Fixed):
        return Fixed(a.val + b.val)

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

    if (ta,tb) == (Fixed, Var):
        return Var(min(a.val, b.lower), max(a.val, b.upper))

    if (ta,tb) == (Var, Fixed):
        return Var(min(a.lower, b.val), max(a.val, b.val))

    if (ta,tb) == (Var, Var):
        return Var(min(a.lower, b.lower), max(b.upper, b.upper))

    # --

    #if (ta,tb) == (Union, Union):
        #return Union(a.parameters + b.parameters)

    # --

    if (ta,tb) == (CType, CType):
        return CType.from_str(promote_types(a.name, b.name).name)

    raise Incommensurable(a,b)
