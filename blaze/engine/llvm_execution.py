import ast

import numpy as np

import numba
from numba import decorators
from numba.ufunc_builder import UFuncBuilder
from numba.minivect import minitypes

import blaze
import blaze.idx
from blaze.expr import visitor
from blaze.expr import ops
from blaze.expr import paterm
from blaze.engine import pipeline
from blaze.engine import executors
from blaze.sources import canonical

from blaze.datashape import datashape
from blaze import Table, NDTable, Array, NDArray

from numbapro.vectorize import Vectorize

class GraphToAst(visitor.BasicGraphVisitor):
    """
    Convert a blaze graph to a Python AST.
    """

    binop_to_astop = {
        ops.Add: ast.Add,
        ops.Mul: ast.Mult,
    }

    def __init__(self):
        super(GraphToAst, self).__init__()
        self.ufunc_builder = UFuncBuilder()

    def App(self, app):
        if app.operator.arity == 2:
            op = binop_to_astop.get(type(app.operator), None)
            if op is not None:
                left, right = self.visit(app.operator.children)
                return ast.BinOp(left=left, op=op(), right=right)

        return self.Unknown(app)

    def Unknown(self, tree):
        return self.ufunc_builder.register_operand(tree)


def build_executor(pyast_function, operands):
    "Build a ufunc and an wrapping executor from a Python AST"
    vectorizer = Vectorize(pyast_function)
    vectorizer.add(*[minitype(op) for op in operands])
    ufunc = vectorizer.build_ufunc()

    operands_dtypes = map(get_dtype, operands)
    # TODO: this should be part of the blaze graph
    result_dtype = reduce(np.promote_types, operands_dtypes)

    # TODO: build an executor tree and substitute where we can evaluate
    executor = executors.ElementwiseLLVMExecutor(
        ufunc,
        operands_dtypes,
        result_dtype,
    )

    return executor

class ATermToAstTranslator(visitor.GraphTranslator):
    """
    Convert an aterm graph to a Python AST.
    """

    opname_to_astop = {
        'add': ast.Add,
        'mul': ast.Mult,
    }

    nesting_level = 0

    def __init__(self, executors):
        super(ATermToAstTranslator, self).__init__()
        self.ufunc_builder = UFuncBuilder()
        self.executors = executors

    def set_executor(self, aterm, executor_id):
        aterm

    def register(self, result):
        if self.nesting_level == 0:
            # Bottom of graph that we can handle
            operands = self.ufunc_builder.operands
            pyast_function = self.ufunc_builder.build_ufunc_ast(result)
            executor = build_executor(pyast_function, operands)
            appl = paterm.AAppl(paterm.ATerm('Executor'), operands)
            appl = paterm.AAnnotation(appl, 'numba', [id(executor)])
            return appl

        self.result = result

        # Delete this node
        return None

    def AAppl(self, app):
        "Look for unops, binops and reductions we can handle"
        if paterm.matches('Arithmetic;*', app.spine):
            opname = app.args[0].lower()
            op = self.opname_to_astop.get(opname, None)
            args = app.args[1:]
            if op is not None and len(args) == 2:
                self.nesting_level += 1
                self.visit(args)
                self.nesting_level -= 1

                left, right = self.result
                result = ast.BinOp(left=left, op=op(), right=right)
                return self.register(result)

        return self.unhandled(app)

    def AInt(self, constant):
        return ast.Num(n=constant.n)

    AFloat = AInt

    def maybe_operand(self, aterm):
        if self.nesting_level:
            self.result = self.ufunc_builder.register_operand(aterm)

    def Array(self, array):
        self.maybe_operand(aterm)
        return array

    def unhandled(self, aterm):
        "An term we can't handle, scan for sub-trees"
        nesting_level = self.nesting_level
        state = self.ufunc_builder.save()

        self.nesting_level = 0
        self.visitchildren(aterm)
        self.nesting_level = nesting_level
        self.ufunc_builder.restore(state)

        self.maybe_operand(aterm)
        return aterm


def getsource(ast):
    from meta import asttools
    return asttools.dump_python_source(ast).strip()

def get_dtype(blaze_obj):
    # This is probably wrong...
    return blaze_obj.datashape.operands[-1].to_dtype()

def minitype(blaze_obj):
    """
    Get the numba/minivect type from a blaze object (NDArray or whatnot).
    This should be a direct mapping. Even better would be to unify the two
    typesystems...
    """
    dtype = get_dtype(blaze_obj)
    return minitypes.map_dtype(dtype)

def substitute_llvm_executors(aterm_graph, executors):
    translator = ATermToAstTranslator(executors)
    return translator.visit(aterm_graph)

if __name__ == '__main__':
    a = NDArray([1, 2, 3, 4], datashape('2, 2, int'))
    ops, funcdef = convert_graph(a + a)
#    print getsource(funcdef)
#    print result

#    import doctest
#    doctest.testmod()
