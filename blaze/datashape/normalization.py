# -*- coding: utf-8 -*-

"""
Datashape normalization. This handles Ellipses and broadcasting.
"""

from functools import partial
from itertools import chain
from collections import defaultdict, deque

from blaze import error
from . import transform, tzip
from .coretypes import (DataShape, Ellipsis, Fixed, CType, Function,
                        TypeVar, int32)

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
    a1, b1 = normalize_datashapes(a, b)
    a2, b2 = normalize_ellipses(a1, b1)
    a3, b3 = normalize_broadcasting(a2, b2)
    return a3, b3

#------------------------------------------------------------------------
# DataShape Normalizers
#------------------------------------------------------------------------

def normalize_datashapes(a, b):
    # Normalize (CType, DataShape) pairs
    if (type(a), type(b)) == (CType, DataShape):
        a = DataShape(a)
    if (type(a), type(b)) == (DataShape, CType):
        b = DataShape(b)

    if (type(a), type(b)) == (DataShape, DataShape):
        return a, b
    return tzip(normalize_datashapes, a, b)


def normalize_ellipses(a, b):
    """
    Normalize ellipses:

        1) '..., T'   : a data shape accepting any number of dimensions
        2) 'A..., T'  : a data shape with a type variable accepting any number
                        of dimensions

    Case 2) needs to be handled specially: Type variable `A` may be used in
    the context several times, and all occurrences must unify, i.e. the
    shapes must broadcast together.
    """
    contexts = {}
    _collect_ellipses_contexts(contexts, a, b)
    partitions = _partition_ellipses_contexts(contexts)
    final_dims, ndims = _broadcast_ellipses_partitions(partitions)
    result1, result2 = _normalize_ellipses(contexts, ndims, a, b)
    return _resolve_ellipsis_return_type(final_dims, result1, result2)


def normalize_broadcasting(a, b):
    """Add broadcasting dimensions to DataShapes"""
    return _normalize_broadcasting(a, b)

#------------------------------------------------------------------------
# Ellipses
#------------------------------------------------------------------------

def _collect_ellipses_contexts(ctx, a, b):
    """
    Collect ellipses contexts for each datashape pair.

    Returns
    =======
    contexts: { (DataShape, DataShape) : { Ellipsis : dimensions } }
        The contexts of the different occurring ellipses.
        `dimensions` is the list of matched dimensions, e.g. [Fixed(10)]
    """
    if (type(a), type(b)) == (DataShape, DataShape):
        result = _normalize_ellipses_datashapes(a, b)
        ctx[a, b] = result
        return a, b
    else:
        return tzip(partial(_collect_ellipses_contexts, ctx), a, b)


def _partition_ellipses_contexts(contexts):
    """
    Partition the ellipses contexts, mapping ellipsis type variables to
    all the matched dimensions.

    Returns
    =======
    partitions: { Ellipses : [dimensions] }
        Map ellipses with type variables to the dimensions they correspond to
    """
    partitions = defaultdict(list)
    for ctx in contexts.itervalues():
        for ellipsis, dims in ctx.iteritems():
            if ellipsis.typevar:
                partitions[ellipsis].append(dims)

    return partitions


def _broadcast_ellipses_partitions(partitions):
    """
    Unify and broadcast all parts of the ellipsis together.

    Returns
    =======
    ndims: { TypeVar : ndim }
        A mapping from each ellipsis type variable to the number of required
        dimensions.
    """
    from .unification import unify_simple

    def dummy_ds(dims):
        return DataShape(*dims + [int32])

    ndims = {}
    final_dims = {}
    for ellipsis, partition in partitions.iteritems():
        final_dshape = reduce(unify_simple, map(dummy_ds, partition))
        final_dims[ellipsis.typevar] = final_dshape.shape
        ndims[ellipsis.typevar] = len(final_dshape.parameters) - 1

    return final_dims, ndims


def _normalize_ellipses(contexts, ndims, a, b):
    """
    Apply the substitution contexts to all the datashapes, filling out any
    missing leading dimensions.
    """
    if (type(a), type(b)) == (DataShape, DataShape):
        S = contexts[a, b]
        for ellipsis, dims in S.iteritems():
            if ellipsis.typevar:
                expected_ndim = ndims[ellipsis.typevar]
                got_ndim = len(dims)
                missing_ndim = (expected_ndim - got_ndim)
                dims[:] = [Fixed(1)] * missing_ndim + dims

        return substitute(S, a), substitute(S, b)
    else:
        return tzip(partial(_normalize_ellipses, contexts, ndims), a, b)

def _resolve_ellipsis_return_type(final_dims, result1, result2):
    """
    Given the broadcast dimensions, and results with ellipses resolved, see
    if we need to manually resolve the return type of functions.

    Arguments
    =========
    final_dims: { TypeVar : [dims] }
        Ellipses with type variables mapping to the final broadcast dimensions
    """
    if (type(result1), type(result2)) == (Function, Function):
        r1 = result1.restype
        r2 = result2.restype
        if isinstance(r1, TypeVar):
            def get_dims(e):
                if isinstance(e, Ellipsis) and e.typevar:
                    return final_dims.get(e.typevar, [e])
                return [e]

            params = list(chain(*[get_dims(e) for e in r2.parameters]))
            r2 = DataShape(*params)
            params = result2.argtypes + (r2,)
            result2 = Function(*params)

    return result1, result2

####--- DataShape ellipsis resolution ---####

def _normalize_ellipses_datashapes(ds1, ds2):
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