"""
Naive type checking
"""

from collections import namedtuple
from itertools import permutations

from ndtable.datashape.coretypes import top
from ndtable.datashape.unification import unify, Incommensurable

typeresult = namedtuple('Types', 'env, dom, cod, opaque')

def dynamic(cls):
    universal = set([top])
    return all(arg == universal for arg in cls.dom)

class InvalidSignature(Exception):
    def __init__(self, args):
        self.args = args
    def __str__(self):
        return "Invalid Signature '%s'" % (self.args)

class TypeCheck(Exception):
    def __init__(self, signature, operant):
        self.signature = signature
        self.operant   = operant

    def __str__(self):
        return 'Signature %s does not permit type %s' % (
            self.signature,
            self.operant,
        )

# Some notes on notation:
#
#     a -> b -> c
#
# Is a function of two arguments, the first of type a, the second
# of type b and returning a type c.
#
# Would be this in Python 3 signature notation:
#
#     def (x : a, y : b) -> c:
#          ...

def typecheck(signature, operands, domc, universe, commutative=False):
    """
    Parameters
        signature : String containing the type signature.
                    Example "a -> b -> a"

        operands  : The operands to type check against signature.

        dom       : The constraints on the space domain to
                    traverse.

        universe  : The universe of terms in which to resolve
                    instances to types.

    Optional
        commutative : Use the commutative checker which attemps to
                      typecheck all permutations of domains to
                      find a satisfiable one.

    Returns:
        env : The enviroment satisfying the given signature
              and operands with the constraints.

    """

    if callable(universe):
        typeof = universe
    else:
        typeof = lambda t: universe[t]

    tokens = [
        tok.strip()
        for tok in
        signature.split('->')
    ]

    dom = tokens[0:-1]
    cod = tokens[-1]

    # Rigid variables are those that appear in multiple times
    # in the signature
    #      a -> b -> a -> c
    #      Rigid a
    #      Free  b,c
    rigid = [tokens.count(token)  > 1 for token in dom]
    free  = [tokens.count(token) == 1 for token in dom]

    #assert len(dom) == self.arity

    env = {}

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
            bound = env.get(var)

            if bound:
                if typeof(operand) != bound:
                    try:
                        uni = unify(operand, bound)
                    except Incommensurable:
                        raise TypeError(
                            'Cannot unify %s %r' % (type(operand), bound))

                    if uni in types:
                        env[var] = uni
            else:
                if typeof(operand) in types:
                    env[var] = typeof(operand)
                else:
                    raise TypeCheck(signature, typeof(operand))


    # Return the unification of the domain and codomain if
    # the signature is satisfiable.

    domt = [env[tok] for tok in dom]
    try:
        codt = env[cod]
        opaque = False
    except KeyError:
        # The codomain is still a free parameter even after
        # unification of the domain, this is normally
        # impossible in Haskell land but we'll allow it here
        # by just returning the top
        codt = top
        opaque = True

    return typeresult(env, domt, codt, opaque)
