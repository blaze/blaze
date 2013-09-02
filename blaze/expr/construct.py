"""
Blaze expression graph construction for deferred evaluation. `construct` is
the entry point to graph construction.
"""

from collections import Iterable

import blaze
from blaze.datashape import coretypes as T
from blaze import Kernel

from .graph import ArrayOp, KernelOp
from .context import ExprContext, unify
from .conf import conf

#------------------------------------------------------------------------
# Graph construction (entry point)
#------------------------------------------------------------------------

def construct(kernel, f, signature, args):
    """
    Parameters
    ----------
    kernel : Blaze Function
        (Overloaded) blaze function representing the operation
    args   : list
        kernel parameters
    """
    assert isinstance(kernel, Kernel)

    params = [] # [(graph_term, ExprContext)]

    # -------------------------------------------------
    # Build type unification parameters

    for i, arg in enumerate(args):
        if isinstance(arg, blaze.Array):
            # Compose new expression using previously constructed expression
            if arg.expr:
                params.append(arg.expr)
                continue
            term = ArrayOp(arg.dshape, arg)
        else:
            term = from_value(arg)

        empty = ExprContext()
        arg.expr = (term, empty)
        params.append(arg.expr)

    # -------------------------------------------------

    assert isinstance(signature, T.Function)
    restype = signature.parameters[-1]

    # -------------------------------------------------

    return KernelOp(restype, *args, kernel=kernel, func=f, signature=signature)

def from_value(value):
    return ArrayOp(T.typeof(value), value)

#------------------------------------------------------------------------
# Argument Munging
#------------------------------------------------------------------------

def is_homogeneous(it):
    # Checks Python types for arguments, not to be confused with the
    # datashape types and the operator types!

    head = it[0]
    head_type = type(head)
    return head, all(type(a) == head_type for a in it)

def injest_iterable(args, depth=0, force_homog=False, conf=conf):
    # TODO: Should be 1 stack frame per each recursion so we
    # don't blow up Python trying to parse big structures
    assert isinstance(args, (list, tuple))

    if depth > conf.max_argument_recursion:
        raise RuntimeError(
        "Maximum recursion depth reached while parsing arguments")

    # tuple, list, dictionary, any recursive combination of them
    if isinstance(args, Iterable):
        if len(args) == 0:
            return []

        if len(args) < conf.max_argument_len:
            sample = args[0:conf.argument_sample]

            # If the first 100 elements are type homogenous then
            # it's likely the rest of the iterable is.
            head, is_homog = is_homogeneous(sample)

            if force_homog and not is_homog:
                raise TypeError("Input is not homogenous.")

            # Homogenous Arguments
            # ====================

            if is_homog:
                return [from_value(a) for a in args]

            # Heterogenous Arguments
            # ======================

            # TODO: This will be really really slow, certainly
            # not something we'd want to put in a loop.
            # Optimize later!

            elif not is_homog:
                ret = []
                for a in args:
                    if isinstance(a, (list, tuple)):
                        sub = injest_iterable(a, depth+1)
                        ret.append(sub)
                    else:
                        ret.append(from_value(a))

                return ret

        else:
            raise RuntimeError("""
            Too many dynamic arguments to build expression
            graph. Consider alternative construction.""")
