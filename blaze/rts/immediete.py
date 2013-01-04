#------------------------------------------------------------------------
# Immediete Evaluation
#------------------------------------------------------------------------

def ieval(fn, args):
    fnargs = []

    for arg in args:
        # TODO: better way of sepcying that it has a "data backend"
        if hasattr(arg, 'data'):
            fnargs.append((arg._datashape, arg.data.read_desc()))
        else:
            fnargs.append(arg)

    return fn(*fnargs)
