"""
Simple kernel selection for execution

- NumPy
- Numexpr
- Blir (experimental)

"""

from astutils import dump
from ast import NodeVisitor

#------------------------------------------------------------------------
# Numexpr
#------------------------------------------------------------------------

from numexpr import evaluate

# --------------------------------

numexpr_kernels = {
    'add' : '%s + %s',
    'mul' : '%s * %s',
     # Express dot product in terms of a product and sum()
    'dot' : 'sum(%s * %s)'
}

# --------------------------------

class NumexprSubtree(NodeVisitor):
    def __init__(self):
        self.vars = {}
        self.expr = ''

    def visit_Kernel(self, node):
        krn = numexpr_kernels[node.name]
        self.expr += krn % tuple(map(self.visit, node.args))

    def visit_Terminal(self, node):
        var = fresh()
        self.vars[var] = node.src
        return var

def eval_numexpr(exp):
    v = NumexprSubtree()
    with namesupply():
        v.visit(exp)

    # this is super naive, just showing that it works
    return evaluate(v.expr, local_dict=v.vars)

#------------------------------------------------------------------------
# Blir
#------------------------------------------------------------------------

from blir import Context, compile, execute

from cgen.blirgen import *
from cgen.kernels import *
from cgen.utils import namesupply

# --------------------------------

blir_kernels = {
    'add'  : '_out0[i0] = _in0[i0] + _in1[i0]',
    'madd' : '_out0[i0] = _in0[i0] + _in1[i0] * _in1[i0] ',
}

blir_typemap = {
    'int64' : 'int',
}

# --------------------------------

class BlirSubtree(NodeVisitor):
    def __init__(self):
        self.args = []
        self.vars = []
        self.kernels = []

    def visit_Kernel(self, node):
        args =  map(self.visit, node.args)

        if node.kind == 1: # ZIPWITH
            with namesupply():
                out = (OUT, args[0][1])
                krn = ElementwiseKernel(args + [out], blir_kernels[node.name])
                self.kernels.append(str(krn))
        else:
            raise NotImplementedError

    def visit_Terminal(self, node):
        if node.ty == 'array':
            arr   = node.src
            size  = np.prod(arr.shape)
            dtype = 'int' #, node.src.dtype
            arg = (IN, VectorArg((size,), 'array[%s]' % dtype))

        elif node.ty == 'scalar':
            typ = blir_typemap[node.ty]
            arg = (IN, ScalarArg('%s' % typ))
        else:
            raise NotImplementedError

        self.vars.append(node.src)
        return arg

# --------------------------------

def eval_blir(exp):
    v = BlirSubtree()
    v.visit(exp)

    # hardcoded against the first kernel for now
    print v.kernels[0]
    ast, env = compile(v.kernels[0])
    ctx = Context(env)

    out = np.empty_like(v.vars[0])
    args = v.vars + [out]

    execute(ctx, args=args, fname='kernel0')
    return out

#------------------------------------------------------------------------
# Python
#------------------------------------------------------------------------

python_kernels = {
    'add' : '+',
    'mul' : '+',
}

def eval_python(exp):
    pass

#------------------------------------------------------------------------
# NumPy
#------------------------------------------------------------------------

import numpy as np

numpy_kernels = {
    'add' : np.add,
    'dot' : np.dot,
    'abs' : np.abs,
    'neg' : np.negative,
    'sin' : np.sin,
    'cos' : np.cos,
}

numpy_ops = {
    '+' : np.add,
    '*' : np.add,
}

numpy_typemap = {
    'int32' : np.int32,
    'int64' : np.int64,
}

class NumpySubtree(NodeVisitor):
    def __init__(self):
        self.kernels = []
        self.args = []

    def visit_Kernel(self, node):
        args = map(self.visit, node.args)

        if node.kind == 1: # ZIPWITH
            pass
        else:
            raise NotImplementedError

    def visit_Terminal(self, node):
        if node.ty == 'array':
            arr = node.src

        elif node.ty == 'scalar':
            typ = blir_typemap[node.ty]
            arg = (IN, ScalarArg('%s' % typ))
        else:
            raise NotImplementedError

        self.vars.append(node.src)
        return arg

def eval_numpy(exp):
    pass
