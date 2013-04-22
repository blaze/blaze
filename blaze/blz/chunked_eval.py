########################################################################
#
#       License: BSD
#       Created: September 10, 2010
#       Author:  Francesc Alted - francesc@continuum.io
#
########################################################################


# Functions for an execution engine for BLZ

def getvars(expression, user_dict, depth, vm):
    """Get the variables in `expression`.

    `depth` specifies the depth of the frame in order to reach local
    or global variables.
    """

    cexpr = compile(expression, '<string>', 'eval')
    if vm == "python":
        exprvars = [ var for var in cexpr.co_names
                     if var not in ['None', 'False', 'True'] ]
    else:
        # Check that var is not a numexpr function here.  This is useful for
        # detecting unbound variables in expressions.  This is not necessary
        # for the 'python' engine.
        exprvars = [ var for var in cexpr.co_names
                     if var not in ['None', 'False', 'True']
                     and var not in numexpr_functions ]


    # Get the local and global variable mappings of the user frame
    user_locals, user_globals = {}, {}
    user_frame = sys._getframe(depth)
    user_locals = user_frame.f_locals
    user_globals = user_frame.f_globals

    # Look for the required variables
    reqvars = {}
    for var in exprvars:
        # Get the value.
        if var in user_dict:
            val = user_dict[var]
        elif var in user_locals:
            val = user_locals[var]
        elif var in user_globals:
            val = user_globals[var]
        else:
            if vm == "numexpr":
                raise NameError("variable name ``%s`` not found" % var)
            val = None
        # Check the value.
        if (vm == "numexpr" and
            hasattr(val, 'dtype') and hasattr(val, "__len__") and
            val.dtype.str[1:] == 'u8'):
            raise NotImplementedError(
                "variable ``%s`` refers to "
                "a 64-bit unsigned integer object, that is "
                "not yet supported in numexpr expressions; "
                "rather, use the 'python' vm." % var )
        if val is not None:
            reqvars[var] = val
    return reqvars


# Assign function `eval` to a variable because we are overriding it
_eval = eval

def eval(expression, vm=None, out_flavor=None, user_dict={}, **kwargs):
    """
    eval(expression, vm=None, out_flavor=None, user_dict={}, **kwargs)

    Evaluate an `expression` and return the result.

    Parameters
    ----------
    expression : string
        A string forming an expression, like '2*a+3*b'. The values for 'a' and
        'b' are variable names to be taken from the calling function's frame.
        These variables may be scalars, barrays or NumPy arrays.
    vm : string
        The virtual machine to be used in computations.  It can be 'numexpr'
        or 'python'.  The default is to use 'numexpr' if it is installed.
    out_flavor : string
        The flavor for the `out` object.  It can be 'barray' or 'numpy'.
    user_dict : dict
        An user-provided dictionary where the variables in expression
        can be found by name.    
    kwargs : list of parameters or dictionary
        Any parameter supported by the barray constructor.

    Returns
    -------
    out : barray object
        The outcome of the expression.  You can tailor the
        properties of this barray by passing additional arguments
        supported by barray constructor in `kwargs`.

    """

    if vm is None:
        vm = ca.defaults.eval_vm
    if vm not in ("numexpr", "python"):
        raiseValue, "`vm` must be either 'numexpr' or 'python'"

    if out_flavor is None:
        out_flavor = ca.defaults.eval_out_flavor
    if out_flavor not in ("barray", "numpy"):
        raiseValue, "`out_flavor` must be either 'barray' or 'numpy'"

    # Get variables and column names participating in expression
    depth = kwargs.pop('depth', 2)
    vars = getvars(expression, user_dict, depth, vm=vm)

    # Gather info about sizes and lengths
    typesize, vlen = 0, 1
    for name in vars.iterkeys():
        var = vars[name]
        if hasattr(var, "__len__") and not hasattr(var, "dtype"):
            raise ValueError, "only numpy/barray sequences supported"
        if hasattr(var, "dtype") and not hasattr(var, "__len__"):
            continue
        if hasattr(var, "dtype"):  # numpy/barray arrays
            if isinstance(var, np.ndarray):  # numpy array
                typesize += var.dtype.itemsize * np.prod(var.shape[1:])
            elif isinstance(var, ca.barray):  # barray array
                typesize += var.dtype.itemsize
            else:
                raise ValueError, "only numpy/barray objects supported"
        if hasattr(var, "__len__"):
            if vlen > 1 and vlen != len(var):
                raise ValueError, "arrays must have the same length"
            vlen = len(var)

    if typesize == 0:
        # All scalars
        if vm == "python":
            return _eval(expression, vars)
        else:
            return ca.numexpr.evaluate(expression, local_dict=vars)

    return eval_blocks(expression, vars, vlen, typesize, vm, out_flavor,
                       **kwargs)

def eval_blocks(expression, vars, vlen, typesize, vm, out_flavor,
                **kwargs):
    """Perform the evaluation in blocks."""

    # Compute the optimal block size (in elements)
    # The next is based on experiments with bench/ctable-query.py
    if vm == "numexpr":
        # If numexpr, make sure that operands fits in L3 chache
        bsize = 2**20  # 1 MB is common for L3
    else:
        # If python, make sure that operands fits in L2 chache
        bsize = 2**17  # 256 KB is common for L2
    bsize //= typesize
    # Evaluation seems more efficient if block size is a power of 2
    bsize = 2 ** (int(math.log(bsize, 2)))
    if vlen < 100*1000:
        bsize //= 8
    elif vlen < 1000*1000:
        bsize //= 4
    elif vlen < 10*1000*1000:
        bsize //= 2
    # Protection against too large atomsizes
    if bsize == 0:
        bsize = 1

    vars_ = {}
    # Get temporaries for vars
    maxndims = 0
    for name in vars.iterkeys():
        var = vars[name]
        if hasattr(var, "__len__"):
            ndims = len(var.shape) + len(var.dtype.shape)
            if ndims > maxndims:
                maxndims = ndims
            if len(var) > bsize and hasattr(var, "_getrange"):
                vars_[name] = np.empty(bsize, dtype=var.dtype)

    for i in xrange(0, vlen, bsize):
        # Get buffers for vars
        for name in vars.iterkeys():
            var = vars[name]
            if hasattr(var, "__len__") and len(var) > bsize:
                if hasattr(var, "_getrange"):
                    if i+bsize < vlen:
                        var._getrange(i, bsize, vars_[name])
                    else:
                        vars_[name] = var[i:]
                else:
                    vars_[name] = var[i:i+bsize]
            else:
                if hasattr(var, "__getitem__"):
                    vars_[name] = var[:]
                else:
                    vars_[name] = var

        # Perform the evaluation for this block
        if vm == "python":
            res_block = _eval(expression, vars_)
        else:
            res_block = ca.numexpr.evaluate(expression, local_dict=vars_)

        if i == 0:
            # Detection of reduction operations
            scalar = False
            dim_reduction = False
            if len(res_block.shape) == 0:
                scalar = True
                result = res_block
                continue
            elif len(res_block.shape) < maxndims:
                dim_reduction = True
                result = res_block
                continue
            # Get a decent default for expectedlen
            if out_flavor == "barray":
                nrows = kwargs.pop('expectedlen', vlen)
                result = ca.barray(res_block, expectedlen=nrows, **kwargs)
            else:
                out_shape = list(res_block.shape)
                out_shape[0] = vlen
                result = np.empty(out_shape, dtype=res_block.dtype)
                result[:bsize] = res_block
        else:
            if scalar or dim_reduction:
                result += res_block
            elif out_flavor == "barray":
                result.append(res_block)
            else:
                result[i:i+bsize] = res_block

    if isinstance(result, ca.barray):
        result.flush()
    if scalar:
        return result[()]
    return result

