# -*- coding: utf-8 -*-

"""
Datashape normalization. This handles Ellipses and broadcasting.
"""

from itertools import chain
from collections import defaultdict, deque

from blaze import error, util
from . import transform, verify
from .coretypes import DataShape, Ellipsis, Fixed, Function

#------------------------------------------------------------------------
# Normalization
#------------------------------------------------------------------------

def normalize(constraints, broadcasting):
    """
    Parameters
    ----------

    constraints : [(DataShape, DataShape)]
        List of constraints (datashape type equations)
    broadcasting: [bool]
        indicates for each constraint whether the two DataShapes broadcast

    Returns: (constraints, broadcast_env)
        A two-tuple containing a list of normalized constraints and a
        broadcasting environment listing all type variables which may
        broadcast together.
    """
    is_ds = lambda ds: isinstance(ds, DataShape)
    result = []
    for ds1, ds2 in constraints:
        if is_ds(ds1) and is_ds(ds2):
            ds1, ds2 = normalize_ellipses(ds1, ds2)
        result.append((ds1, ds2))

    constraints, b_env = normalize_broadcasting(result, broadcasting)
    return constraints, b_env

#------------------------------------------------------------------------
# Ellipses

def normalize_ellipses(a, b):
    S = _normalize_ellipses(a, b)
    return substitute(S, a), substitute(S, b)

def _normalize_ellipses(ds1, ds2):
    if not (isinstance(ds1, DataShape) and isinstance(ds2, DataShape)):
        return

    # -------------------------------------------------
    # Find ellipses

    a = [x for x in  ds1.parameters if isinstance(x, Ellipsis)]
    b = [x for x in  ds2.parameters if isinstance(x, Ellipsis)]
    xs, ys = list(ds1.parameters[-2::-1]), list(ds2.parameters[-2::-1])

    # -------------------------------------------------
    # Match ellipses

    if a and (len(xs) <= len(ys) or not b):
        S = match(xs, ys)
    elif b and (len(ys) <= len(xs) or not a):
        S = match(ys, xs)
    elif a or b:
        assert len(xs) == len(ys)
        S = match(xs, ys)
    else:
        return ds1, ds2 # no ellipses, nothing to do

    # -------------------------------------------------
    # Reverse the reversed matches

    for x, L in S.iteritems():
        S[x] = L[::-1]

    # -------------------------------------------------
    # Error checking

    if a and b:
        # We have an ellipsis in either operand. We mandate that one
        # 'contains' the other, since it is unclear how to unify them if
        # they are disjoint
        [x], [y] = a, b
        if x not in S[y] and y not in S[x]:
            raise error.BlazeTypeError(
                "Unable to line up Ellipses in %s and %s" % (ds1, ds2))

        if not S[x]:
            S[x].append(y)
        if not S[y]:
            S[y].append(x)

    return S

def match(xs, ys, S=None):
    if S is None:
        S = defaultdict(list)

    xs, ys = deque(xs), deque(ys)
    while xs and ys:
        x = xs.popleft()
        if isinstance(x, Ellipsis):
            while len(ys) > len(xs):
                S[x].append(ys.popleft())
        else:
            y = ys.popleft()
            if isinstance(y, Ellipsis):
                S[y].append(x)
                xs, ys = ys, xs # match(ys, xs, S)

    return S

def substitute(S, ds):
    """Substitute a solution mapping Elipses to parameters"""
    sub_param = lambda x: S[x] if isinstance(x, Ellipsis) else [x]
    return DataShape(*chain(*map(sub_param, ds.parameters)))


#------------------------------------------------------------------------
# Broadcasting

def normalize_broadcasting(constraints, broadcasting):
    result = []        # [(DataShape, DataShape)]
    broadcast_env = [] # [(typevar1, typevar2)]

    for broadcast, (ds1, ds2) in zip(broadcasting, constraints):
        if broadcast and (isinstance(ds1, DataShape) and
                          isinstance(ds2, DataShape)):
            # Create type variables for leading dimensions
            len1, len2 = len(ds1.parameters), len(ds2.parameters)
            leading = tuple(Fixed(1) for i in range(abs(len1 - len2)))

            if len1 < len2:
                ds1 = DataShape(*leading + ds1.parameters)
            elif len2 < len1:
                ds2 = DataShape(*leading + ds2.parameters)

            broadcast_env.extend(zip(ds1.parameters[:-1], ds2.parameters[:-1]))

        result.append((ds1, ds2))

    return result, broadcast_env


#------------------------------------------------------------------------
# Simplification
#------------------------------------------------------------------------

def simplify(t, solution):
    """
    Simplify constraints by eliminating Implements (e.g. '10, A : numeric') and
    type variables associated with Ellipsis (e.g. 'A..., int32'), and by
    updating the given typing solution.

    Parameters
    ----------
    t : Mono
        Blaze type

    Returns: Mono
        Simplified blaze type
    """
    return transform(Simplifier(solution), t)


class Simplifier(object):
    """Simplify a type and update a typing solution"""

    def __init__(self, S):
        self.S = S

    def Implements(self, term):
        typeset = self.S.setdefault(term.typevar, set())
        typeset.add(term.typeset)
        # typeset.update(term.typeset)
        return term.typevar

    # TODO: ellipsis
    # def Ellipsis(self, term):
    #     if term.typevar:
    #         typeset = self.S.setdefault(term.typevar, set()
    #         typeset.update(term)
    #         return term.typevar
    #     return term