# -*- coding: utf-8 -*-

"""
This module implements unification of datashapes. Unification is a general
problem that solves a system of equations between terms. In our case, the
terms are types (datashapes).

A difference with conventional unification is that our equations are not
necessarily looking for equality, they must account for coercions and
broadcasting.

A normalization phase accounts for broadcasting missing leading dimensions
by padding the types with one-sized dimensions. Coercion is resolved through
coercion constraints which act like subset relations. A unifier is
accompanied by a set of constraints but must hold for the free variables in
that type.
"""

from itertools import chain
from functools import partial

from blaze import error
from blaze.util import IdentityDict, IdentitySet
from blaze.datashape.coretypes import (Mono, DataShape, TypeVar, promote, free,
                                       type_constructor, IntegerConstant,
                                       StringConstant, CType, Fixed)

#------------------------------------------------------------------------
# Entry point
#------------------------------------------------------------------------

def unify(constraints, broadcasting):
    """
    Unify a set of constraints and return a concrete solution

        >>> import blaze
        >>> d1 = blaze.dshape('10, int32')
        >>> d2 = blaze.dshape('T, float32')
        >>> [result], constraints = unify([(d1, d2)], [True])
        >>> result
        dshape("10, float32")
        >>> constraints
        []
    """
    # Compute solution to a set of constraints
    constraints, b_env = normalize(constraints, broadcasting)
    solution, remaining = unify_constraints(constraints)
    resolve_typsets(remaining, solution)

    # Compute a type substitution with concrete types from the solution
    # TODO: incorporate broadcasting environment during reification
    substitution = reify(solution)

    # Reify and promote the datashapes
    result = [substitute(substitution, ds2) for ds1, ds2 in constraints]
    return result, [(a, solution[b]) for a, b in remaining]

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
    solve by generating new coercion constraints in the form of
    [(A coerces to B)], where A and B are type variables.

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
        remaining = []
        for t1, t2 in constraints:
            for freevar in chain(free(t1), free(t2)):
                solution[freevar] = set()

    for t1, t2 in constraints:
        unify_single(t1, t2, solution, remaining)

    return solution, remaining

def unify_single(t1, t2, solution, remaining):
    """
    Unify a single type equation and update the solution and remaining
    constraints.
    """
    if isinstance(t1, TypeVar) and isinstance(t2, TypeVar):
        remaining.append((t1, t2))
    elif isinstance(t1, TypeVar):
        if t1 in free(t2):
            raise error.UnificationError("Cannot unify recursive types")
        solution[t1].add(t2)
    elif isinstance(t2, TypeVar):
        unify_single(t2, t1, solution, remaining)
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
            unify_single(arg1, arg2, solution, remaining)


def resolve_typsets(constraints, solution):
    """
    Fix-point iterate until all type variables are resolved according to the
    constraints.

    Parameters
    ----------
    constraints: [(TypeVar, TypeVar)]
        Constraint graph specifying coercion relations:
            (a, b) ∈ constraints with a ⊆ b

    solution : { TypeVar : set([ Type ]) }
        Typing solution
    """
    # Fix-point update our type sets
    changed = True
    while changed:
        changed = False
        for a, b in constraints:
            length = len(solution[b])
            solution[b].update(solution[a])
            changed |= length < len(solution[b])

    # Update empty type set solutions with variables from the input
    # We can do this since there is no contraint on the free variable
    empty = IdentitySet()
    for a, b in constraints:
        if not solution[b] or b in empty:
            solution[b].add(a)
            empty.add(b)

    # # Fix-point our fix-point
    # if empty:
    #     resolve_typsets(constraints, solution)

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
        freevars = IdentityDict.fromkeys(chain(*[free(t) for t in typeset]))

        if not typeset:
            S[typevar] = typevar
            typeset.add(typevar)
            continue
        elif freevars:
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
        # Promote CTypes
        return promote(*units)


def substitute(solution, ds):
    """
    Substitute a typing solution for a type, resolving all free type variables.
    """
    if isinstance(ds, TypeVar):
        return solution[ds] or ds
    elif not isinstance(ds, Mono) or isinstance(ds, CType):
        return ds
    else:
        typecon = type_constructor(ds)
        return typecon(*[substitute(solution, p) for p in ds.parameters])


if __name__ == '__main__':
    import doctest
    doctest.testmod()