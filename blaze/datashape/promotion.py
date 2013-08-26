# -*- coding: utf-8 -*-

"""
This module implements type coercion rules for data shapes.

Note that transitive coercions could be supported, but we decide not to since
it may involve calling a whole bunch of functions with a whole bunch of types
to figure out whether this is possible in the face of polymorphic overloads.
"""

from blaze import error
from blaze.datashape.coretypes import (DataShape, IntegerConstant,
                                       StringConstant, CType, Fixed,
                                       to_numpy)

import numpy as np

#------------------------------------------------------------------------
# Type unit promotion
#------------------------------------------------------------------------

def promote_units(*units):
    """
    Promote unit types, which are either CTypes or Constants.
    """
    unit = units[0]
    if len(units) == 1:
        return unit
    elif isinstance(unit, Fixed):
        assert all(isinstance(u, Fixed) for u in units)
        if len(set(units)) > 2:
            raise error.UnificationError(
                "Got multiple differing integer constants", units)
        else:
            left, right = units
            if left == IntegerConstant(1):
                return right
            elif right == IntegerConstant(1):
                return left
            else:
                if left != right:
                    raise error.UnificationError(
                        "Cannot unify differing fixed dimensions "
                        "%s and %s" % (left, right))
                return left
    elif isinstance(unit, StringConstant):
        for u in units:
            if u != unit:
                raise error.UnificationError(
                    "Cannot unify string constants %s and %s" % (unit, u))

        return unit

    else:
        # Promote CTypes -- this should go through coerce()
        return promote(*units)

def promote(a, b):
    """Promote a series of CType or DataShape types"""
    if isinstance(a, DataShape):
        assert isinstance(b, DataShape)
        assert a.parameters[:-1] == b.parameters[:-1]
        return DataShape(a.parameters[:-1] + (promote(a.measure, b.measure),))

    return CType.from_numpy_dtype(np.result_type(to_numpy(a), to_numpy(b)))
