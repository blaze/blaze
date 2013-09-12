# -*- coding: utf-8 -*-

"""
Type promotion.
"""

from itertools import product
from functools import reduce

from blaze import error
from blaze.util import gensym
from blaze.datashape import (DataShape, CType, Fixed, to_numpy,
                             TypeSet, TypeVar)

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

    # -------------------------------------------------
    # Fixed

    if isinstance(a, Fixed):
        assert isinstance(b, Fixed)
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
        # Promote CTypes -- this should go through coerce()
        return promote_scalars(a, b)

    # -------------------------------------------------
    # DataShape

    elif isinstance(a, (DataShape, CType)) and isinstance(b, (DataShape, CType)):
        return promote_datashapes(a, b)

    else:
        raise TypeError("Unknown types, cannot promote: %s and %s" % (a, b))


def eq(a, b):
    if isinstance(a, TypeVar) and isinstance(b, TypeVar):
        return True
    return a == b

def promote_scalars(a, b):
    """Promote two CTypes"""
    return CType.from_numpy_dtype(np.result_type(to_numpy(a), to_numpy(b)))

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
