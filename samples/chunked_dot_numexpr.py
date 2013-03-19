"""

Example that evaluates an expression like `(x+y).dot(a*z + b*w)`, where
x, y, z and w are vector that live on disk.  The computation is carried out
by using numexpr.  The code here first massages first the expression above to
achieve something like `sum((x+y)*(a*z + b*w))` that can easily be computed
by numexpr.

Usage:

$ chunked_dot_numexpr create  # creates the vectors on-disk
$ chunked_dot_numexpr run     # computes the expression
$ chunked_dot_numexpr delete  # removes the vectors from disk

"""

import blaze
import blaze.blir as blir
import numpy as np
import math
from time import time

_locals = None
_params = None

# Machinery for evaluating expressions via python or numexpr
# ---------------

def _getvars(expression, user_dict, depth, vm):
    """Get the variables in `expression`.

    `depth` specifies the depth of the frame in order to reach local
    or global variables.
    """

    cexpr = compile(expression, '<string>', 'eval')
    if vm == "python":
        exprvars = [ var for var in cexpr.co_names
                     if var not in ['None', 'False', 'True'] ]
    else:
        import numexpr
        # Check that var is not a numexpr function here.  This is useful for
        # detecting unbound variables in expressions.  This is not necessary
        # for the 'python' engine.
        from numexpr.expressions import functions as numexpr_functions
        exprvars = [ var for var in cexpr.co_names
                     if var not in ['None', 'False', 'True']
                     and var not in numexpr_functions ]


    # Get the local and global variable mappings of the user frame
    user_locals, user_globals = {}, {}
    # user_frame = sys._getframe(depth)
    # user_locals = user_frame.f_locals
    # user_globals = user_frame.f_globals

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


def evaluate(expression, vm=None, out_flavor=None, user_dict={}, **kwargs):
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

    if vm is None:
        vm = "python"
    if vm not in ("numexpr", "python"):
        raiseValue, "`vm` must be either 'numexpr' or 'python'"

    if out_flavor is None:
        out_flavor = "blaze"
    if out_flavor not in ("blaze", "numpy"):
        raiseValue, "`out_flavor` must be either 'blaze' or 'numpy'"

    # Get variables and column names participating in expression
    depth = kwargs.pop('depth', 2)
    vars = _getvars(expression, user_dict, depth, vm=vm)

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


class Operation(object):
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def __add__(self, rhs):
        return Operation('+', self, rhs)

    def __sub__(self, rhs):
        return Operation('-', self, rhs)

    def __mul__(self, rhs):
        return Operation('*', self, rhs)

    def dot(self, rhs):
        return Operation('dot', self, rhs)

    def eval(self, locals=None, params=None):
        global _locals, _params
        _locals, _params = locals, params
        return evaluate(str(self), vm=params['vm'], user_dict=locals)

    def __str__(self):
        if self.op == "dot":
            # Re-express dot product in terms of a product and an sum()
            return 'sum' + "(" + str(self.lhs) + "*" + str(self.rhs) + ")"
        else:
            return "(" + str(self.lhs) + str(self.op) + str(self.rhs) + ")"

    def __repr__(self):
        return ('Operation(' + repr(self.op) + ', ' 
                + repr(self.lhs) + ', ' 
                + repr(self.rhs) + ')')
        

class Terminal(object):
    def __init__(self, src):
        self.source = src

    def __add__(self, rhs):
        return Operation('+', self, rhs)

    def __sub__(self, rhs):
        return Operation('-', self, rhs)

    def __mul__(self, rhs):
        return Operation('*', self, rhs)

    def dot(self, rhs):
        return Operation('dot', self, rhs)

    def __str__(self):
        return self.source

    def __repr__(self):
        return 'Terminal(' + repr(self.source) + ')'


# ================================================================

_persistent_array_names = ['chunk_sample_x.blz', 
                           'chunk_sample_y.blz', 
                           'chunk_sample_z.blz',
                           'chunk_sample_w.blz']

def _create_persistent_array(name, dshape):
    print 'creating ' + name + '...'
    blaze.ones(dshape, params=blaze.params(storage=name, clevel=0))

def _delete_persistent_array(name):
    from shutil import rmtree
    rmtree(name)

def create_persistent_arrays(args):
    elements = args[0] if len(args) > 0 else '10000000'
    dshape = elements + ', float64'

    try:
        dshape = blaze.dshape(dshape)
    except:
        print elements + ' is not a valid size for the arrays'
        return

    for name in _persistent_array_names:
        _create_persistent_array(name, dshape)

def delete_persistent_arrays():
    for name in _persistent_array_names:
        _delete_persistent_array(name)

def run_test(args):
    T = Terminal

    x = T('x')
    y = T('y')
    z = T('z')
    w = T('w')
    a = T('a')
    b = T('b')
    vm = "python" if "python" in args else "numexpr"
    print "evaluating expression with '%s' vm..." % vm
    expr = (x+y).dot(a*z + b*w)

    print 'opening blaze arrays...'
    x_ = blaze.open(_persistent_array_names[0])
    y_ = blaze.open(_persistent_array_names[1])
    z_ = blaze.open(_persistent_array_names[2])
    w_ = blaze.open(_persistent_array_names[3])
    a_ = 2.0
    b_ = 2.0

    if 'in_memory' in args:
        print 'getting an in-memory version of blaze arrays...'
        params = blaze.params(clevel=0)
        t0 = time()
        x_ = blaze.array(x_[:], params=params)
        y_ = blaze.array(y_[:], params=params)
        z_ = blaze.array(z_[:], params=params)
        w_ = blaze.array(w_[:], params=params)
        print "conversion to blaze in-memory: %.3f" % (time() - t0)

    print 'datashape is:', x_.datashape

    if 'print_expr' in args:
        print expr

    t_ce = time()
    expr_vars = {'x': x_, 'y': y_, 'z': z_, 'w': w_, 'a': a_, 'b': b_, }
    result_ce = expr.eval(expr_vars, params={'vm': vm})
    t_ce = time() - t_ce
    print "'%s' vm result is : %s in %.3f s" % (vm, result_ce, t_ce)
    
    # in numpy...
    print 'evaluating expression with numpy...'
    x_ = x_[:]
    y_ = y_[:]
    z_ = z_[:]
    w_ = w_[:]

    t_np = time()
    result_np = np.dot(x_+y_, a_*z_ + b_*w_)
    t_np = time() - t_np

    print 'numpy result is : %s in %.3f s' % (result_np, t_np)


def main(args):
    command = args[1] if len(args) > 1 else 'help'

    if command == 'create':
        create_persistent_arrays(args[2:])
    elif command == 'run':
        run_test(args)
    elif command == 'delete':
        delete_persistent_arrays()
    else:
        print args[0] + ' [create elements|run|delete]' 

if __name__ == '__main__':
    from sys import argv
    main(argv)


## Local Variables:
## mode: python
## coding: utf-8 
## python-indent: 4
## tab-width: 4
## fill-column: 66
## End:
