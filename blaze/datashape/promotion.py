# -*- coding: utf-8 -*-

"""
Type promotion.
"""

from itertools import product
from functools import reduce

from blaze import error
from .util import gensym, verify
from .typesets import TypeSet, TypeVar, TypeConstructor
from .coretypes import DataShape, CType, Fixed, Var, to_numpy

import numpy as np

#------------------------------------------------------------------------
# Type unit promotion
#------------------------------------------------------------------------

def promote_units(*units):
    """
    Promote unit types, which are either CTypes or Constants.
    """
    return reduce(promote, units)

def promote(a, b):
    """Promote two blaze types"""

    if a == b:
        return a

    # -------------------------------------------------
    # Fixed

    elif isinstance(a, Fixed):
        if isinstance(b, Fixed):
            if a == Fixed(1):
                return b
            elif b == Fixed(1):
                return a
            else:
                if a != b:
                    raise error.UnificationError(
                        "Cannot unify differing fixed dimensions "
                        "%s and %s" % (a, b))
                return a
        elif isinstance(b, Var):
            if a == Fixed(1):
                return b
            else:
                return a
        else:
            raise TypeError("Unknown types, cannot promote: %s and %s" % (a, b))

    # -------------------------------------------------
    # Var

    elif isinstance(a, Var):
        if isinstance(b, Fixed):
            if b == Fixed(1):
                return a
            else:
                return b
        elif isinstance(b, Var):
            return a
        else:
            raise TypeError("Unknown types, cannot promote: %s and %s" % (a, b))

    # -------------------------------------------------
    # Typeset

    elif isinstance(a, TypeSet) and isinstance(b, TypeSet):
        # TODO: Find the join in the lattice with the below as a fallback ?
        return TypeSet(*set(promote(t1, t2)
                                for t1, t2 in product(a.types, b.types)))

    elif isinstance(a, TypeSet):
        if b not in a.types:
            raise error.UnificationError(
                "Type %s does not belong to typeset %s" % (b, a))
        return b

    elif isinstance(b, TypeSet):
        return promote(b, a)

    # -------------------------------------------------
    # Units

    elif isinstance(a, CType) and isinstance(b, CType):
        # Promote CTypes -- this should use coercion_cost()
        return promote_scalars(a, b)

    # -------------------------------------------------
    # DataShape

    elif isinstance(a, (DataShape, CType)) and isinstance(b, (DataShape, CType)):
        return promote_datashapes(a, b)

    elif (isinstance(type(a), TypeConstructor) and
              isinstance(type(b), TypeConstructor)):
        return promote_type_constructor(a, b)

    else:
        raise TypeError("Unknown types, cannot promote: %s and %s" % (a, b))


def eq(a, b):
    if isinstance(a, TypeVar) and isinstance(b, TypeVar):
        return True
    return a == b

def promote_scalars(a, b):
    """Promote two CTypes"""
    try:
        return CType.from_numpy_dtype(np.result_type(to_numpy(a), to_numpy(b)))
    except TypeError as e:
        raise TypeError("Cannot promote %s and %s: %s" % (a, b, e))

def promote_datashapes(a, b):
    """Promote two DataShapes"""
    from .unification import unify
    from .normalization import normalize_simple

    # Normalize to determine parameters (eliminate broadcasting, etc)
    a, b = normalize_simple(a, b)
    n = len(a.parameters[:-1])

    # Allocate dummy result type for unification
    dst = DataShape(*[TypeVar(gensym()) for i in range(n + 1)])

    # Unify
    [result1, result2], _ = unify([(a, dst), (b, dst)], [True, True])

    assert result1 == result2
    return result1

def promote_type_constructor(a, b):
    """Promote two generic type constructors"""
    # Verify type constructor equality
    verify(a, b)

    # Promote parameters according to flags
    args = []
    for flag, t1, t2 in zip(a.flags, a.parameters, b.parameters):
        if flag['coercible']:
            result = promote(t1, t2)
        else:
            if t1 != t2:
                raise error.UnificationError(
                    "Got differing types %s and %s for unpromotable type "
                    "parameter in constructors %s and %s" % (t1, t2, a, b))
            result = t1

        args.append(result)

    return type(a)(*args)
