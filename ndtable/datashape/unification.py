"""
Unification is a generalization of Numpy broadcasting.

In Numpy we two arrays and broadcast them to yield similar shaped
object::

    A      (2d array):  5 x 4
    B      (1d array):      1
    Result (2d array):  5 x 4

In Blaze we take two arrays with more complex datashapes and
unify them at each coordinate.

    A                :  3, 2, Var(3), int32
    B                :     1, Var(4), float32
    Result           :  3, 2, Var(4), float32

Where this is decomposed coordinate wise.

      3 | 2 | Var(3) | int32

    +   | 1 | Var(4) | float32
    ---------------------------

      unify( Fixed(2), Fixed(0) ) = Fixed(3)
    = 3

      unify( Fixed(2), Fixed(1) ) = Fixed(2)

          2

      unify( Var(3), Var(4) ) = Var(4)

               Var(4)

      unify( int32, float32 ) = float32

                       float32

    = 3, 2, Var(4), float32

"""

from numpy import promote_types
from coretypes import Fixed, Var, TypeVar, Record, \
    CType, top, dynamic

class Incommensurable(Exception):
    def __init__(self, space, dim):
        self.space = space
        self.dim   = dim

    def __str__(self):
        return "No way of unifying (%s) (%s)" % (
            self.space, self.dim
        )

# TODO: Deprecated
def union(dim1, dim2):
    """
    General case
    """
    x  , y  = dim1[0]  , dim2[0]
    xs , ys = dim1[1:] , dim2[1:]

    z = unify(x,y)

    if xs and ys:
        return union(xs, ys)*z
    else:
        return z

def unify(context, a, b):
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
