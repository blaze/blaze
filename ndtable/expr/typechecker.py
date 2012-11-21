"""
Naive type inference for Blaze expressions.

Some notes on notation::

Type System
-----------

A collection of three functions::

    unifier :: ty -> ty -> ty
    typeof :: value -> ty

And a collection of type objects with two special terms::

    (?)   - dynamic type
    (top) - top type

This is specified by a namedtuple of the form::

    typesystem = namedtuple('TypeSystem', 'unifier, top, dynamic, typeof')

Dynamic
-------

A dynamic type written as ( ? ). It allows explicit down casting
and upcasting from any type to any type. In Blaze we use this to
represent opaque types that afford no specialization.

Signatures
----------

The signature::

    a -> b -> c

Stands for a function of two arguments, the first of type a, the second
of type b and returning a type c.

Would be this in Python 3 signature notation::

    def (x : a, y : b) -> c:
        pass

Rigid & Free
------------

Rigid variables are those that appear in multiple times
in the signature::

     f : a -> b -> a -> c

     Rigid : [a]
     Free  : [b,c]

Context
-------

A context records the lexically-bound variables during the progression
of the type checking algorithm. It is a stateful object like a memo
passed through the unifiers.

The context is usually written as :math:`\Gamma` in the literature.

"""

from collections import namedtuple
from itertools import permutations

from ndtable.datashape import coretypes
from ndtable.datashape.unification import Incommensurable

#------------------------------------------------------------------------
# Type Check Exceptions
#------------------------------------------------------------------------

class InvalidSignature(Exception):
    def __init__(self, args):
        self.args = args
    def __str__(self):
        return "Invalid Signature '%s'" % (self.args)

class TypeCheck(Exception):
    def __init__(self, sig, ty):
        self.sig = sig
        self.ty  = ty

    def __str__(self):
        return 'Signature %s does not permit type %s' % (
            self.signature,
            self.operant,
        )

#------------------------------------------------------------------------
# System Specification
#------------------------------------------------------------------------

typesystem = namedtuple('TypeSystem', 'unifier, top, dynamic, typeof')
typeresult = namedtuple('Satisifes', 'env, dom, cod, dynamic')

#------------------------------------------------------------------------
# Core Typechecker
#------------------------------------------------------------------------

def emptycontext():
    return {'top': top, 'dynamic': dynamic}

def dynamic(cls):
    universal = set([coretypes.top])
    return all(arg == universal for arg in cls.dom)

def tyconsist(context, ty1, ty2):
    raise NotImplementedError

def tyeqv(context, ty1, ty2):
    # Right now this uses the __eq__ method defined on the types
    # from expr.coretypes which is kind of ugly... factor this
    # out later.
    return ty1 == ty2

def simplifyty(context, ty):
    return ty

def tyeval(signature, operands, domc, system, commutative=False):
    """

    Parameters:

        signature:
            String containing the type signature.

        operands:
            The operands to type check against signature.

        dom:
            The constraints on the space domain to traverse.

        universe:
            The universe of terms in which to resolve instances to types.

        commutative:
            Use the commutative checker which attempts to eval all
            permutations of domains to find a satisfiable one.

    Returns:

        The context satisfying the given signature and operands with
        the constraints.

    """
    top       = system.top
    unify     = system.unifier

    if callable(system.typeof):
        typeof = system.typeof
    else:
        typeof = lambda t: system.typeof[t]

    # Commutative type checker can be written in terms of an enumeration
    # of the flat tyeval over the permutations of the operands and
    # domain constraints.
    if commutative:
        # TODO: write this better after more coffee
        for p in permutations(zip(operands, domc), 2):
            ops = [q[0] for q in p] # operators, unzip'd
            dcs = [q[1] for q in p] # domain constraints, unzip'd
            try:
                return tyeval(signature, ops, dcs, system, commutative=False)
            except TypeCheck:
                continue
        raise TypeCheck(signature, operands)

    tokens = [
        tok.strip()
        for tok in
        signature.split('->')
    ]

    dom = tokens[0:-1]
    cod = tokens[-1]

    # f : a -> b -> a -> c
    #
    # Rigid : [a]
    # Free  : [b,c]

    rigid = [tokens.count(token)  > 1 for token in dom]
    free  = [tokens.count(token) == 1 for token in dom]

    context = {}

    dom_vars = dom

    # Naive Type Checker
    # ==================

    for i, (var, types, operand) in enumerate(zip(dom_vars, domc, operands)):

        if free[i]:
            # Need only satisfy the kind constraint
            if typeof(operand) not in types:
                raise TypeCheck(signature, typeof(operand))

        if rigid[i]:
            # Need to satisfy the kind constraint and be
            # unifiable in the enviroment context of the
            # other rigid variables.
            bound = context.get(var)

            if bound:
                if typeof(operand) != bound:
                    try:
                        uni = unify(context, typeof(operand), bound)
                    except Incommensurable:
                        raise TypeError(
                            'Cannot unify %s %r' % (typeof(operand), bound))

                    if uni in types:
                        context[var] = uni
                    else:
                        raise TypeError(
                            'Cannot unify %s %r' % (typeof(operand), bound))
            else:
                if typeof(operand) in types:
                    context[var] = typeof(operand)
                else:
                    raise TypeCheck(signature, typeof(operand))


    # Return the unification of the domain and codomain if
    # the signature is satisfiable.

    domt = [context[tok] for tok in dom]
    try:
        codt = context[cod]
        dynamic = False
    except KeyError:
        # The codomain is still a free parameter even after
        # unification of the domain, this is normally
        # impossible in Haskell land but we'll allow it here
        # by just returning the top
        codt = top
        dynamic = True

    return typeresult(context, domt, codt, dynamic)
