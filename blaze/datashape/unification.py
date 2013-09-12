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

import logging
from collections import deque
from itertools import chain

from blaze import error
from blaze.py2help import dict_iteritems, _strtypes
from blaze.util import IdentityDict, IdentitySet
from blaze.datashape import (promote_units, normalize, simplify, tmap,
                             dshape, verify)
from blaze.datashape.coretypes import TypeVar, free

logger = logging.getLogger(__name__)

#------------------------------------------------------------------------
# Entry points
#------------------------------------------------------------------------

def unify_simple(a, b):
    """Unify two blaze types"""
    if isinstance(a, _strtypes):
        a = dshape(a)
    if isinstance(b, _strtypes):
        b = dshape(b)
    [res], _ = unify([(a, b)], [True])
    return res

def unify(constraints, broadcasting=None):
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
    S = IdentityDict()
    constraints = [(simplify(ds1, S), simplify(ds2, S))
                        for ds1, ds2 in constraints]

    # Compute a solution to a set of constraints
    constraints, b_env = normalize(constraints, broadcasting)
    logger.debug("Normalized constraints: %s", constraints)

    solution, remaining = unify_constraints(constraints, S)
    logger.debug("Initial solution: %s", solution)

    seed_typesets(remaining, solution)
    merge_typevar_sets(remaining, solution)

    # Compute a type substitution with concrete types from the solution
    # TODO: incorporate broadcasting environment during reification
    substitution = reify(solution)
    logger.debug("Substitution: %s", substitution)

    # Reify and promote the datashapes
    result = [substitute(substitution, ds2) for ds1, ds2 in constraints]
    return result, remaining

#------------------------------------------------------------------------
# Unification
#------------------------------------------------------------------------

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
    constraints : [(Mono, Mono)]
        List of constraints (blaze type equations)

    Returns: { TypeVar : set([ Mono ]) }
        Returns a solution to the set of constraints. The solution is a set
        of bindings (a substitution) from type variables to type sets.
    """
    solution = IdentityDict(solution)
    remaining = []

    # Initialize solution
    for t1, t2 in constraints:
        for freevar in chain(free(t1), free(t2)):
            if freevar not in solution:
                solution[freevar] = set()

    # Calculate solution
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
        remaining.append((t1, t2))
    elif isinstance(t2, TypeVar):
        if t2 in free(t1):
            raise error.UnificationError("Cannot unify recursive types")
        solution[t2].add(t1)
    elif not free(t1) and not free(t2):
        # No need to recurse, this will be caught by promote()
        pass
    else:
        verify(t1, t2)
        for arg1, arg2 in zip(t1.parameters, t2.parameters):
            unify_single(arg1, arg2, solution, remaining)


def seed_typesets(constraints, solution):
    """
    Resolve type sets by seeding empty sets with type variables.
    """
    # Update empty type set solutions with variables from the input
    # We can do this since there is no constraint on the free variable
    empty = IdentitySet()
    for a, b in constraints:
        if not solution.get(b) or b in empty:
            solution.setdefault(b, IdentitySet()).add(a)
            empty.add(b)

def merge_typevar_sets(constraints, solution):
    """
    Update  the solution of sets that contain only type variables with a new
    type variable along with new coercion constraints.

    Consider the example:

        x, y, z = dshapes('A, B, int32', 'C, D, float32', 'X, Y, float32')

    with x ⊆ z and y ⊆ z:

        >>> A, B, C, D, X, Y = map(TypeVar, 'ABCDXY')
        >>> constraints = [(A, X), (C, X), (B, Y), (D, Y)]
        >>> solution = {X: set([A, C]), Y: set([B, D])}
        >>> merge_typevar_sets(constraints, solution)
        >>> solution[X]
        set([TypeVar(A_C)])
        >>> solution[Y]
        set([TypeVar(B_D)])
    """
    for src_var, typeset in list(dict_iteritems(solution)):
        if len(typeset) > 1 and all(isinstance(v, TypeVar) for v in typeset):
            new_var = TypeVar("_".join(sorted(v.symbol for v in typeset)))
            solution[src_var] = set([new_var])
            solution[new_var] = set()
            for v in typeset:
                constraints.append((v, new_var))
                constraints.remove((v, src_var))

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

    seen = set()
    queue = deque(dict_iteritems(solution))
    while queue:
        typevar, t = queue.popleft()
        t = frozenset(t)
        if typevar in S:
            continue

        typeset = solution[typevar]
        freevars = IdentityDict.fromkeys(chain(*[free(t) for t in typeset]))

        if not typeset:
            S[typevar] = typevar
            typeset.add(typevar)
            continue
        elif freevars and (typevar, t) not in seen:
            # Reify dependencies first
            queue.append((typevar, t))
            seen.add((typevar, t))
        elif freevars:
            typeset = set(substitute(S, t) for t in typeset)

        S[typevar] = promote_units(*typeset)

    return S

#------------------------------------------------------------------------
# Substitution
#------------------------------------------------------------------------

def substitute(solution, ds):
    """
    Substitute a typing solution for a type, resolving all free type variables.
    """
    def f(t):
        if isinstance(t, TypeVar):
            return solution[t] or t
        return t
    return tmap(f, ds)


if __name__ == '__main__':
    import doctest
    doctest.testmod()