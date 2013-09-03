# -*- coding: utf-8 -*-

"""
Type promotion.
"""

from itertools import product

from blaze import error
from blaze.datashape import (DataShape, IntegerConstant, StringConstant,
                             CType, Fixed, to_numpy, TypeSet, TypeVar)

import numpy as np

#------------------------------------------------------------------------
# Type unit promotion
#------------------------------------------------------------------------

def promote_units(*units):
    """
    Promote unit types, which are either CTypes or Constants.
    """
    return reduce(_promote_units, units)

def _promote_units(a, b):
    if isinstance(a, Fixed):
        assert isinstance(b, Fixed)
        if a == IntegerConstant(1):
            return b
        elif b == IntegerConstant(1):
            return a
        else:
            if a != b:
                raise error.UnificationError(
                    "Cannot unify differing fixed dimensions "
                    "%s and %s" % (a, b))
            return a

    elif isinstance(a, StringConstant):
        if a != b:
            raise error.UnificationError(
                "Cannot unify string constants %s and %s" % (a, b))

        return a

    elif isinstance(a, TypeSet) and isinstance(b, TypeSet):
        # TODO: Find the join in the lattice with the below as a fallback ?
        return TypeSet(*set(_promote_units(t1, t2)
                                for t1, t2 in product(a.types, b.types)))

    elif isinstance(a, TypeSet):
        if b not in a.types:
            raise error.UnificationError(
                "Type %s does not belong to typeset %s" % (b, a))
        return b

    elif isinstance(b, TypeSet):
        return _promote_units(b, a)

    else:
        # Promote CTypes -- this should go through coerce()
        return promote(a, b)

def eq(a, b):
    if isinstance(a, TypeVar) and isinstance(b, TypeVar):
        return True
    return a == b

def promote(a, b):
    """Promote a series of CType or DataShape types"""
    if isinstance(a, DataShape):
        assert isinstance(b, DataShape)
        assert all(eq(p1, p2) for p1, p2 in zip(a.parameters[:-1],
                                                b.parameters[:-1]))
        return DataShape(*a.parameters[:-1] + (promote(a.measure, b.measure),))

    return CType.from_numpy_dtype(np.result_type(to_numpy(a), to_numpy(b)))
