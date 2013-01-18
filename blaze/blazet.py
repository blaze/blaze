"""
BlazeT consists of of the extensible set of Blaze datashape types and
all expression types.
"""

from blaze.datashape.coretypes import int_, float_, string, top, dynamic
from blaze.expr.graph import App, Fun, ArrayNode, IntNode, FloatNode, StringNode

# Type checking and unification
from blaze.datashape.unification import unify
from blaze.expr.typechecker import typesystem

#------------------------------------------------------------------------
# Deconstructors
#------------------------------------------------------------------------

def typeof(obj):
    """
    BlazeT value deconstructor, maps values to types. Only
    defined for Blaze types.

    >>> typeof(IntNode(3))
    int64
    >>> typeof(Any())
    top
    >>> typeof(NDArray([1,2,3]))
    dshape("3, int64")
    """
    typ = type(obj)

    # -- special case --
    if isinstance(obj, ArrayNode):
        return obj.datashape

    if typ is App:
        return obj.cod
    elif typ is Fun:
        return obj.cod
    elif typ is IntNode:
        return int_
    elif typ is FloatNode:
        return float_
    elif typ is StringNode:
        return string
    elif typ is dynamic:
        return top
    else:
        raise TypeError, type(obj)

#------------------------------------------------------------------------
# Blaze Typesystem
#------------------------------------------------------------------------

# unify   : the type unification function
# top     : the top type
# dynamic : the dynamic type
# typeof  : the value deconstructor

# Judgements over a type system are uniquely defined by three things:
#
# * a type unifier
# * a top type
# * a value deconstructor
# * first order terms

BlazeT = typesystem(unify, top, any, typeof)
