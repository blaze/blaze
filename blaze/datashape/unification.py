"""
This module implements unification of datashapes. Unification is a general
problem that solves a system of equations between terms. In our case, the
terms are types (datashapes).

A difference with conventional unification is that our equations are not
necessarily looking for equality, they must account for coercions and
broadcasting. What we do to address this is a normalization phase that
normalizes the types so they are acceptable for unification.

We only allow type variables to be shared in a single "signature". This is
naturally enforced by the datashape grammar:

    int32_2d   = T, T, int32
    float64_2d = T, T, float64

In these type specification T is used to specify a local constraint, namely
that it is a two-dimensional square array. However, the T's in the different
datashapes specify different type variables that merely happen to carry the
same name.
"""

from itertools import chain
from functools import partial

from blaze import error
from blaze.util import IdentityDict
from blaze.datashape.coretypes import (Mono, DataShape, TypeVar, promote, free,
                                       type_constructor, IntegerConstant,
                                       StringConstant, CType)

#------------------------------------------------------------------------
# Entry point
#------------------------------------------------------------------------

def unify(constraints, broadcasting):
    """
    Unify a set of constraints and return a concrete solution

        >>> import blaze
        >>> d1 = blaze.dshape('10, int32')
        >>> d2 = blaze.dshape('T, float32')
        >>> [result] = unify([(d1, d2)], [True])
        >>> result
        dshape("10, float64")
    """
    # Compute solution to a set of constraints
    constraints, b_env = normalize(constraints, broadcasting)
    solution = unify_constraints(constraints)

    # Compute a type substitute with concrete types from the solution
    # TODO: incorporate broadcasting environment during reification
    substitution = reify(solution)

    # Reify and promote the datashapes
    sub = partial(substitute, substitution)
    return [promote_units(sub(ds1), sub(ds2)) for ds1, ds2 in constraints]

#------------------------------------------------------------------------
# Unification
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
    result = []        # [(DataShape, DataShape)]
    broadcast_env = [] # [(typevar1, typevar2)]

    for broadcast, (ds1, ds2) in zip(broadcasting, constraints):
        if broadcast:
            # Create type variables for leading dimensions
            len1, len2 = len(ds1.parameters), len(ds2.parameters)
            leading = tuple(TypeVar('Broadcasting%d' % i)
                                for i in range(abs(len1 - len2)))

            if len1 < len2:
                ds1 = DataShape(leading + ds1.parameters)
            elif len2 < len1:
                ds2 = DataShape(leading + ds2.parameters)

            broadcast_env.extend(zip(ds1.parameters, ds2.parameters))

        result.append((ds1, ds2))

    return result, broadcast_env


def unify_constraints(constraints, solution=None):
    """
    Blaze type unification. Two types unify if:

        - They have the same type constructor, and
        - They have an equal number of sub-terms which unify element-wise

    Our algorithm is different from a conventional implementation in that we
    have a different notion of equality since we allow coercion which we
    solve by tracking sets of types. E.g. if we have

        eq 0: T1 = int32
        eq 1: T1 = T2
        eq 2: T1 = float32

    We cannot substitute int32 for T1 in the remaining constraints, since
    that would result in the equation `int32 = float32`, which is clearly
    wrong since we recorded { T1 : int32 } as a solution. Instead we obtain
    successsively in three steps:

        start: solution = { T1: set([]), T2: set([]) }
        ----------------------------------------------
        step0: solution = { T1: set([int32]), T2: set([]) }
        step1: solution = { T1: set([int32]), T2: set([int32]) }
        step2: solution = { T1: set([int32, float32]), T2: set([int32, float32]) }

    Equation 2 updates the type of type variable T2 since the type is shared
    between T1 and T2.

    Parameters
    ----------
    constraints : [(DataShape, DataShape)]
        List of constraints (datashape type equations)

    Returns: { TypeVar : set([ Mono ]) }
        Returns a solution to the set of constraints. The solution is a set
        of bindings (a substitution) from type variables to type sets.
    """
    if solution is None:
        solution = IdentityDict()
        for t1, t2 in constraints:
            for freevar in chain(free(t1), free(t2)):
                solution[freevar] = set()

    for t1, t2 in constraints:
        unify_single(t1, t2, solution)

    return solution

def unify_single(t1, t2, solution):
    """
    Unify a single type equation and update the solution and remaining
    constraints.
    """
    if isinstance(t1, TypeVar) and isinstance(t2, TypeVar):
        solution[t1] = solution[t2] = solution[t1] | solution[t2]
    elif isinstance(t1, TypeVar):
        if t1 in free(t2):
            raise error.UnificationError("Cannot unify recursive types")
        solution[t1].add(t2)
    elif isinstance(t2, TypeVar):
        unify_single(t2, t1, solution)
    elif not free(t1) and not free(t2):
        # No need to recurse, this will be caught by promote()
        pass
    else:
        if not isinstance(t1, Mono) or not isinstance(t2, Mono):
            if t1 != t2:
                raise error.UnificationError("%s != %s" % (t1, t2))
            return

        args1, args2 = t1.parameters, t2.parameters
        tcon1, tcon2 = type_constructor(t1), type_constructor(t2)

        if tcon1 != tcon2:
            raise error.UnificationError(
                "Got differing type constructors %s and %s" % (tcon1, tcon2))

        if len(args1) != len(args2):
            raise error.UnificationError("%s got %d and %d arguments" % (
                            tcon1, len(args1), len(args2)))

        for arg1, arg2 in zip(args1, args2):
            unify_single(arg1, arg2, solution)


def reify(solution, S=None):
    """
    Reify a typing solution, returning a new solution with types as concrete
    types as opposed to type sets.

    Parameters
    ----------
    solution : { TypeVar : set([ Type ]) }
        Typing solution

    Returns: { TypeVar : Type }
        Returns a solution reduced to concrete types only.
    """
    if S is None:
        S = IdentityDict()

    for typevar, t in solution.iteritems():
        if typevar in S:
            continue

        typeset = solution[typevar]
        freevars = set(chain(*[free(t) for t in typeset]))
        if freevars:
            # Reify dependencies first
            reify(dict((v, solution[v]) for v in freevars), S)
            typeset = set(substitute(S, t) for t in typeset)

        S[typevar] = promote_units(*typeset)

    return S


def promote_units(*units):
    """
    Promote unit types, which are either CTypes or Constants.
    """
    unit = units[0]
    if len(units) == 1:
        return unit
    elif isinstance(unit, IntegerConstant):
        assert all(isinstance(u, IntegerConstant) for u in units)
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
                assert left == right
                return left
    elif isinstance(unit, StringConstant):
        for u in units:
            if u != unit:
                raise error.UnificationError(
                    "Cannot unify string constants %s and %s" % (unit, u))

        return unit

    else:
        # Promote CTypes
        return promote(*units)


def substitute(solution, ds):
    """
    Substitute a typing solution for a type, resolving all free type variables.
    """
    if isinstance(ds, TypeVar):
        return solution[ds]
    elif not isinstance(ds, Mono) or isinstance(ds, CType):
        return ds
    else:
        typecon = type_constructor(ds)
        return typecon(*[substitute(solution, p) for p in ds.parameters])


if __name__ == '__main__':
    import doctest
    doctest.testmod()