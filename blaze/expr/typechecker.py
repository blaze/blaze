from collections import namedtuple
from itertools import permutations

from blaze.datashape import coretypes
from blaze.expr.signatures import sig_parse
from blaze.datashape.unification import Incommensurable

#------------------------------------------------------------------------
# Type Check Exceptions
#------------------------------------------------------------------------

class InvalidTypes(TypeError):
    def __init__(self, sig, ty):
        self.sig = sig
        self.ty  = ty

    def __str__(self):
        return 'Signature %s does not permit type %s' % (
            self.sig,
            self.ty,
        )

#------------------------------------------------------------------------
# System Specification
#------------------------------------------------------------------------

typesystem    = namedtuple('TypeSystem', 'unifier, top, dynamic, typeof')

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def emptycontext(system):
    topt = system.top
    dynt = system.dynamic
    return {'top': topt, 'dynamic': dynt}


def infer(signature, expression, constrs, system, commutative=False):
    """ Type inference

    Parameters
    ----------
    signature : str,
        String containing the type signature.

    expression : object
        expression to resolve types of

    constrs : dict
        The constraints/bounds on the typevars signature.

    system : TypeSystem
        The type system over which to evaluate.

    commutative : bool
        Use the commutative checker which attempts to eval all
        permutations of domains in commutative operators to
        find a judgement.

    Returns
    -------
        The context satisfying the given signature and operands with
        the constraints.

    """

    # unpack the named tuple
    unify  = system.unifier
    topt   = system.top
    dynt   = system.dynamic
    typeof = system.typeof

    # Commutative checker can be written in terms of an enumeration of
    # the flat tyeval over the permutations of the operands and domain
    # constraints.
    if commutative:
        raise NotImplementedError

    tokens = sig_parse(signature)

    dom = tokens[0]
    cod = tokens[1]

    rigid = [tokens.count(token) >  1 for token in dom + cod]
    free  = [tokens.count(token) == 1 for token in dom + cod]

    context = emptycontext(system)

#------------------------------------------------------------------------
# ATerm Deconstructors
#------------------------------------------------------------------------

# Add :: (a,a) -> a
#
# Add(
#   Array(39558864){dshape("x int64")}
# , Array(39558864){dshape("y int64")}
# )
#
# Yields Constraints:
#
#  x = { ? }
#  y = { x, 1 }

def aterm_typeof(aterm):
    return aterm.annotation.ty
