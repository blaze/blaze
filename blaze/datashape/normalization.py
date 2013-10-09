# -*- coding: utf-8 -*-

"""
Datashape normalization. This handles Ellipses and broadcasting.
"""

from functools import partial
from itertools import chain
from collections import defaultdict, deque

from blaze import error
from . import transform, tzip
from .coretypes import DataShape, Ellipsis, Fixed, CType

#------------------------------------------------------------------------
# Normalization
#------------------------------------------------------------------------

def normalize(constraints, broadcasting=None):
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
    broadcasting_env = None
    result = [normalize_simple(a, b) for a, b in constraints]
    return result, broadcasting_env

def normalize_simple(a, b):
    ellipsis_ctx = {}

    def _normalize_simple(a, b):
        # Normalize (CType, DataShape) pairs
        if (type(a), type(b)) == (CType, DataShape):
            a = DataShape(a)
        if (type(a), type(b)) == (DataShape, CType):
            b = DataShape(b)

        # Normalize ellipses and broadcasting
        if (type(a), type(b)) == (DataShape, DataShape):
            if (len(a.parameters), len(b.parameters)) != (1, 1):
                (a, b) = normalize_ellipses(a, b, ellipsis_ctx)
            a, b = normalize_broadcasting(a, b)
        else:
            a, b = tzip(_normalize_simple, a, b)

        return a, b

    return _normalize_simple(a, b)

#------------------------------------------------------------------------
# DataShape Normalizers
#------------------------------------------------------------------------

def normalize_ellipses(a, b, solution={}):
    """Eliminate ellipses in DataShape"""
    S = _normalize_ellipses(a, b)
    check_ellipsis_consistency(S, solution)
    result = substitute(S, a), substitute(S, b)
    solution.update(S)
    return result

def normalize_broadcasting(a, b):
    """Add broadcasting dimensions to DataShapes"""
    return _normalize_broadcasting(a, b)

#------------------------------------------------------------------------
# Ellipses
#------------------------------------------------------------------------

def _normalize_ellipses(ds1, ds2):
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
        return {} # no ellipses, nothing to do

    # -------------------------------------------------
    # Reverse the reversed matches

    for x, L in S.items():
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


def check_ellipsis_consistency(solutions1, solutions2):
    """
    Check that all ellipsis type variables has valid dimensions within the
    given solution contexts.
    """
    # Collect all typevars in A..., int32 etc
    ellipsis_typevars = {}
    for ellipsis, dims in chain(solutions1.iteritems(), solutions2.iteritems()):
        if not ellipsis.typevar:
            continue

        seen_ndim = ellipsis_typevars.get(ellipsis.typevar)
        if seen_ndim is not None and len(dims) != seen_ndim:
            # TODO: Broadcasting?
            raise error.UnificationError(
                "Differing dimensionality for type variable %s, got %s "
                "and %s" % (ellipsis.typevar, seen_ndim, len(dims)))

        ellipsis_typevars[ellipsis.typevar] = len(dims)

#------------------------------------------------------------------------
# Broadcasting
#------------------------------------------------------------------------

def _normalize_broadcasting(a, b):
    if isinstance(a, DataShape) and isinstance(b, DataShape):
        # Create type variables for leading dimensions
        len1, len2 = len(a.parameters), len(b.parameters)
        leading = tuple(Fixed(1) for i in range(abs(len1 - len2)))

        if len1 < len2:
            a = DataShape(*leading + a.parameters)
        elif len2 < len1:
            b = DataShape(*leading + b.parameters)
    else:
        a, b = tzip(_normalize_broadcasting, a, b)

    return a, b

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