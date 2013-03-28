""" Provides the evaluator backends for Numexpr and the Python interpreter """


from expression_builder import Visitor
import blaze
import math

def evaluate(expression, vm='python', out_flavor='blaze', user_dict={}, **kwargs):
    """
    evaluate(expression, vm=None, out_flavor=None, user_dict=None, **kwargs)

    Evaluate an `expression` and return the result.

    Parameters
    ----------
    expression : string
        A string forming an expression, like '2*a+3*b'. The values for 'a' and
        'b' are variable names to be taken from the calling function's frame.
        These variables may be scalars, carrays or NumPy arrays.
    vm : string
        The virtual machine to be used in computations.  It can be 'numexpr'
        or 'python'.  The default is to use 'numexpr' if it is installed.
    out_flavor : string
        The flavor for the `out` object.  It can be 'Blaze' or 'numpy'.
    user_dict : dict
        An user-provided dictionary where the variables in expression
        can be found by name.    
    kwargs : list of parameters or dictionary
        Any parameter supported by the carray constructor.

    Returns
    -------
    out : Blaze object
        The outcome of the expression.  You can tailor the
        properties of this Blaze array by passing additional arguments
        supported by carray constructor in `kwargs`.

    """

    if vm not in ('numexpr', 'python'):
        raiseValue, "`vm` must be either 'numexpr' or 'python'"

    if out_flavor not in ('blaze', 'numpy'):
        raiseValue, "`out_flavor` must be either 'blaze' or 'numpy'"

    # Get variables and column names participating in expression
    vars = user_dict

    # Gather info about sizes and lengths
    typesize, vlen = 0, 1
    for name in vars.iterkeys():
        var = vars[name]
        if not hasattr(var, "datashape"):
            # scalar detection
            continue
        else:  # blaze arrays
            shape, dtype = blaze.to_numpy(var.datashape)
            typesize += dtype.itemsize
            lvar = shape[0]
            if vlen > 1 and vlen != lvar:
                raise ValueError, "arrays must have the same length"
            vlen = lvar

    if typesize == 0:
        # All scalars
        if vm == "python":
            return eval(expression, vars)
        else:
            import numexpr
            return numexpr.evaluate(expression, local_dict=vars)

    return _eval_blocks(expression, vars, vlen, typesize, vm, out_flavor,
                        **kwargs)

def _eval_blocks(expression, vars, vlen, typesize, vm, out_flavor,
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
        if hasattr(var, "datashape"):
            shape, dtype = blaze.to_numpy(var.datashape)
            ndims = len(shape) + len(dtype.shape)
            if ndims > maxndims:
                maxndims = ndims

    for i in xrange(0, vlen, bsize):
        # Get buffers for vars
        for name in vars.iterkeys():
            var = vars[name]
            if hasattr(var, "datashape"):
                shape, dtype = blaze.to_numpy(var.datashape)
                vars_[name] = var[i:i+bsize]
            else:
                if hasattr(var, "__getitem__"):
                    vars_[name] = var[:]
                else:
                    vars_[name] = var

        # Perform the evaluation for this block
        if vm == "python":
            res_block = eval(expression, None, vars_)
        else:
            import numexpr
            res_block = numexpr.evaluate(expression, local_dict=vars_)

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
            if out_flavor == "blaze":
                nrows = kwargs.pop('expectedlen', vlen)
                result = blaze.array(res_block, **kwargs)
            else:
                out_shape = list(res_block.shape)
                out_shape[0] = vlen
                result = np.empty(out_shape, dtype=res_block.dtype)
                result[:bsize] = res_block
        else:
            if scalar or dim_reduction:
                result += res_block
            elif out_flavor == "blaze":
                result.append(res_block)
            else:
                result[i:i+bsize] = res_block

    # if isinstance(result, blaze.Array):
    #     result.flush()
    if scalar:
        return result[()]
    return result

# End of machinery for evaluating expressions via python or numexpr
# ---------------

class _ExpressionBuilder(Visitor):
    def accept_operation(self, node):
        str_lhs = self.accept(node.lhs)
        str_rhs = self.accept(node.rhs)
        if node.op == "dot":
            # Re-express dot product in terms of a product and an sum()
            return 'sum' + "(" + str_lhs + "*" + str_rhs + ")"
        else:
            return "(" + str_lhs + node.op + str_rhs + ')'

    def accept_terminal(self, node):
        return node.source


# ================================================================            

class NumexprEvaluator(object):
    """ Evaluates expressions using numexpr """
    name = 'numexpr'

    def __init__(self, root_node, operands=None):
        assert(operands)
        self.str_expr = _ExpressionBuilder().accept(root_node)
        self.operands = operands

    def eval(self, chunk_size=None):
        return evaluate(self.str_expr,
                        vm='numexpr',
                        user_dict=self.operands)
    

class NumpyEvaluator(object):
    name = 'python interpreter with numpy'
    def __init__(self, root_node, operands=None):
        assert(operands)
        self.str_expr = _ExpressionBuilder().accept(root_node)
        self.operands = operands

    def eval(self, chunk_size=None):
        return evaluate(self.str_expr, 
                        vm='python',
                        user_dict=self.operands)

