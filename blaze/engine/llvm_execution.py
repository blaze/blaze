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
from blaze.datashape import coretypes
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

    def register(self, graph, result):
        if self.nesting_level == 0:
            # Bottom of graph that we can handle
            operands = self.ufunc_builder.operands
            pyast_function = self.ufunc_builder.build_ufunc_ast(result)
            py_ufunc = self.ufunc_builder.compile_to_pyfunc(pyast_function)

            executor = build_executor(py_ufunc, operands, graph)
            self.executors[id(executor)] = executor
            annotation = paterm.AAnnotation('numba', [id(executor)])
            appl = paterm.AAppl(paterm.ATerm('Executor'), operands,
                                annotation=annotation)
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
                return self.register(app, result)

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


def build_executor(py_ufunc, operands, aterm_subgraph_root):
    "Build a ufunc and an wrapping executor from a Python AST"
    result_dtype = get_dtype(aterm_subgraph_root)
    operand_dtypes = map(get_dtype, operands)

    vectorizer = Vectorize(py_ufunc)
    vectorizer.add(restype=minitype(result_dtype),
                   argtypes=map(minitype, operand_dtypes))
    ufunc = vectorizer.build_ufunc()

    # TODO: build an executor tree and substitute where we can evaluate
    executor = executors.ElementwiseLLVMExecutor(
        ufunc,
        operand_dtypes,
        result_dtype,
    )

    return executor

def getsource(ast):
    from meta import asttools
    return asttools.dump_python_source(ast).strip()

def get_dtype(aterm):
    type_repr = aterm.annotation['type']
    dshape = datashape(type_repr.s)
    dtype = coretypes.to_dtype(dshape)
    return dtype

def minitype(dtype):
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
