#------------------------------------------------------------------------
# Immediete Evaluation
#------------------------------------------------------------------------

def ieval(fn, args):
    fnargs = [(a._datashape, a.data.read_desc()) for a in args]

    return 42
    #return fn(fnargs)
