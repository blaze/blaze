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
from blaze import error
from .coretypes import DataShape, TypeVar, promote, free, type_constructor

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
            len1, len2 = len(ds1.params), len(ds2.params)
            leading = tuple(TypeVar('Broadcasting%d' % i)
                                for i in range(abs(len1 - len2)))

            if len1 < len2:
                ds1 = DataShape(leading + ds1.parameters)
            elif len2 < len1:
                ds2 = DataShape(leading + ds2.parameters)

            broadcast_env.extend(zip(ds1.parameters, ds2.parameters))

        result.append((ds1, ds2))

    return result, broadcast_env


def unify(constraints, solution=None):
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
        solution = {}
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
            pass
        solution[t1].add(t2)
    elif isinstance(t2, TypeVar):
        unify_single(t2, t1, solution)
    else:
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
        S = {}

    for typevar, t in solution.iteritems():
        if typevar in S:
            continue

        typeset = solution[typevar]
        freevars = set(chain(free(t) for t in typeset))
        if freevars:
            # Reify dependencies first
            reify(dict((v, solution[v]) for v in freevars), S)
            typeset = set(substitute(S, t) for t in typeset)

        S[typevar] = promote(*typeset)

    return S


def substitute(solution, ds):
    """
    Substitute a typing solution for a type, resolving all free type variables.
    """
    if isinstance(ds, TypeVar):
        return solution[ds]
    else:
        typecon = type_constructor(ds)
        return typecon(*[substitute(solution, p) for p in ds.parameters])