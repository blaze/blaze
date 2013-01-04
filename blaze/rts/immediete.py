#------------------------------------------------------------------------
# Immediete Evaluation
#------------------------------------------------------------------------

from blaze.expr.graph import ArrayNode

def ieval(fn, args):
    fnargs = []

    for arg in args:
        if isinstance(arg, ArrayNode):
            fnargs.append(arg._datashape, arg.data.read_desc())
        else:
            fnargs.append(arg)

    return fn(*fnargs)
