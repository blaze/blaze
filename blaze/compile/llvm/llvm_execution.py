import ast
import inspect
from functools import partial

import numpy as np

import numba
from numba import transforms as numba_transforms
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
from blaze.engine import numba_kernels, numba_reductions
from blaze.sources import canonical
from blaze import plan

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


def is_something(name, app):
    if isinstance(app, paterm.AAppl):
        app = app.spine
    return paterm.matches('%s;*' % name, app)

is_arithmetic   = partial(is_something, 'Arithmetic')
is_math         = partial(is_something, 'Math')
is_reduction    = partial(is_something, 'Reduction')
is_full_reduction = is_reduction
is_slice        = partial(is_something, 'Slice')
is_assign       = partial(is_something, 'Assign')
is_array        = partial(is_something, 'Array')
is_none         = partial(is_something, 'None')

class ATermToAstTranslator(visitor.GraphTranslator):
    """
    Convert an aterm graph to a Python AST.
    """

    opname_to_astop = {
        'add': ast.Add,
        'mul': ast.Mult,
    }

    opname_to_reduce_kernel = dict(inspect.getmembers(numba_reductions,
                                                      inspect.isfunction))

    nesting_level = 0

    def __init__(self, executors, blaze_operands):
        super(ATermToAstTranslator, self).__init__()
        self.ufunc_builder = UFuncBuilder()
        self.executors = executors
        self.blaze_operands = blaze_operands # term -> blaze graph object

    def get_blaze_op(self, term):
        term_id = term.annotation.meta[0].label
        return self.blaze_operands[term_id]

    def strategy(self, graph, operands):
        return "chunked"

    def build_executor(self, graph, operands, result):
        result_dtype = unannotate_dtype(graph)
        strategy = self.strategy(graph, operands)

        if is_reduction(graph):
            operator, operand = graph.args
            opname = operator.label.lower()
            kernel = self.opname_to_reduce_kernel[opname]

            executor = executors.NumbaFullReducingExecutor(
                    strategy, kernel, minitype(result_dtype), operation=opname)
            fillvalue = graph.annotation.meta[1]
        else:
            pyast_function = self.ufunc_builder.build_ufunc_ast(result)
            # print getsource(pyast_function)
            py_ufunc = self.ufunc_builder.compile_to_pyfunc(pyast_function,
                                                            globals={'np': np})

            operand_dtypes = map(unannotate_dtype, operands)
            executor = build_ufunc_executor(operand_dtypes, py_ufunc,
                                            pyast_function, result_dtype,
                                            strategy)
            fillvalue = None

        return executor, fillvalue

    def register_operand(self, graph, result, lhs):
        operands = self.ufunc_builder.operands

        executor, fillvalue = self.build_executor(graph, operands, result)
        self.executors[id(executor)] = executor

        if lhs is not None:
            operands.append(lhs)
            datashape = lhs.annotation.ty #self.get_blaze_op(lhs).datashape
        else:
            # blaze_operands = [self.get_blaze_op(op) for op in operands]
            # datashape = coretypes.broadcast(*blaze_operands)
            datashape = graph.annotation.ty

        annotation = paterm.AAnnotation(
            ty=datashape,
            annotations=[id(executor), 'numba', bool(lhs), fillvalue]
        )
        appl = paterm.AAppl(paterm.ATerm('Executor'), operands,
                            annotation=annotation)
        return appl

    def register(self, graph, result, lhs=None):
        if lhs is not None:
            assert self.nesting_level == 0

        if self.nesting_level == 0:
            # Bottom of graph that we can handle
            return self.register_operand(graph, result, lhs)

        self.result = result

        # Delete this node
        return None

    def match_assignment(self, app):
        """
        Handles slice assignemnt, e.g. out[:, :] = non_trivial_expr
        """
        assert self.nesting_level == 0

        lhs, rhs = app.args

        #--------------------------------------------------------------------
        # Visit RHS
        #--------------------------------------------------------------------
        self.nesting_level += 1
        self.visit(rhs)
        rhs_result = self.result
        self.nesting_level -= 1

        #--------------------------------------------------------------------
        # Visit LHS
        #--------------------------------------------------------------------
        is_simple = (is_slice(lhs) and is_array(lhs.args[0]) and
                     all(is_none(v) for v in lhs.args[1:]))
        if is_simple:
            self.nesting_level += 1
            lhs = self.visit(lhs)
            self.nesting_level -= 1
            lhs = self.ufunc_builder.operands.pop() # pop LHS from operands
        else:
            # LHS is complicated, let someone else (or ourselves!) execute
            # it independently
            # self.nesting_level is 0 at this point, so it will be registered
            # independently
            state = self.ufunc_builder.save()
            lhs = self.visit(lhs)
            lhs_result = self.result
            self.ufunc_builder.restore(state)

        #--------------------------------------------------------------------
        # Build and return kernel if the rhs was an expression we could handle
        #--------------------------------------------------------------------
        if rhs_result:
            return self.register(app, rhs_result, lhs=lhs)
        else:
            app.args = [lhs, rhs]
            return app

    def handle_math_or_arithmetic(self, app):
        """
        Rewrite math and arithmetic operations.
        """
        opname = app.args[0].label.lower()
        if is_arithmetic(app):
            op = self.opname_to_astop.get(opname, None)
        else:
            # TODO: unhack
            if hasattr(np, opname):
                op = opname
            else:
                op = None

        type = plan.get_datashape(app)

        # Only accept scalars if we are already nested
        is_array = type.shape or self.nesting_level

        if op and is_array: # args = [op, ...]
            self.nesting_level += 1
            self.visit(app.args[1:])
            self.nesting_level -= 1

            # handle_arithmetic/handle_math
            if is_arithmetic(app):
                return self.handle_arithmetic(app, op)
            else:
                return self.handle_math(app, op)

        return self.unhandled(app)

    def handle_arithmetic(self, app, ast_op):
        """
        Handle unary and binary arithmetic
        """
        left, right = self.result
        result = ast.BinOp(left=left, op=ast_op(), right=right)
        return self.register(app, result)

    def handle_math(self, app, math_func_name):
        """
        Handle math calls by generate a call like `np.sin(x)`
        """
        operand = self.result
        func = ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),
                             attr=math_func_name,
                             ctx=ast.Load())
        math_call = ast.Call(func=func, args=[operand], keywords=[],
                             starargs=None, kwargs=None)
        return self.register(app, math_call)

    def handle_full_reduction(self, app):
        """
        Handle things like (A * B).sum().
        """
        # TODO: Fuse potential numba kernel operand with reduction operation

        operator, operand = app.args
        self.visitchildren0(app)
        self.ufunc_builder.operands.append(operand)
        return self.register_operand(app, None, None)

    def AAppl(self, app):
        "Look for unops, binops and reductions and anything else we can handle"
        if is_arithmetic(app) or is_math(app):
            return self.handle_math_or_arithmetic(app)

        if is_full_reduction(app):
            return self.handle_full_reduction(app)

        elif is_slice(app):
            array, start, stop, step = app.args
            if all(is_none(op) for op in (start, stop, step)):
                return self.visit(array)

        elif is_assign(app):
            return self.match_assignment(app)

        elif is_array(app) and self.nesting_level:
            self.maybe_operand(app)
            if self.nesting_level:
                return None
            return app

        return self.unhandled(app)

    def AInt(self, constant):
        self.result = ast.Num(n=constant.n)
        return constant

    AFloat = AInt

    def maybe_operand(self, aterm):
        if self.nesting_level:
            self.result = self.ufunc_builder.register_operand(aterm)

    def visitchildren0(self, aterm):
        """
        Visit the children of a term at nesting level 0, and restore the
        level afterwards. This is useful to register a potential numba
        subtree as a separate (or unhandled) operand, e.g.:

            (A * B).sum()
               ^
               |___ register as separate operand
        """
        nesting_level = self.nesting_level
        self.nesting_level = 0
        self.visitchildren(aterm)
        self.nesting_level = nesting_level

    def unhandled(self, aterm):
        "An term we can't handle, scan for sub-trees"
        state = self.ufunc_builder.save()
        self.visitchildren0(aterm)
        self.ufunc_builder.restore(state)
        self.maybe_operand(aterm)
        return aterm


def build_ufunc_executor(operand_dtypes, py_ufunc, pyast_function, result_dtype,
                         strategy):

    vectorizer = Vectorize(py_ufunc)
    vectorizer.add(restype=minitype(result_dtype),
                   argtypes=map(minitype, operand_dtypes))
    ufunc = vectorizer.build_ufunc()

    # Get a string of the operation for debugging
    return_stat = pyast_function.body[0]
    operation = getsource(return_stat.value)

    executor = executors.ElementwiseLLVMExecutor(
        strategy,
        ufunc,
        operand_dtypes,
        result_dtype,
        operation=operation,
        )
    return executor

def build_executor(py_ufunc, pyast_function, operands,
                   aterm_subgraph_root, strategy='chunked'):
    """ Build a ufunc and an wrapping executor from a Python AST """
    result_dtype = unannotate_dtype(aterm_subgraph_root)
    operand_dtypes = map(unannotate_dtype, operands)

    executor = build_ufunc_executor(operand_dtypes, py_ufunc, pyast_function,
                                    result_dtype, strategy)

    return executor

def getsource(ast):
    from meta import asttools
    return asttools.dump_python_source(ast).strip()

def unannotate_dtype(aterm):
    """ Takes a term with a datashape annotation and returns the NumPy
    dtype associate with it

    >>> term
    x{dshape("2, 2, int32")}
    >>> unannotate_dtype(term)
    int32
    """
    # unpack the annotation {'s': 'int32'}
    unpack = paterm.matches('dshape(s);*', aterm.annotation['type'])
    ds_str = unpack['s']
    dshape = datashape(ds_str.s)

    dtype = coretypes.to_dtype(dshape)
    return dtype

def minitype(dtype):
    return minitypes.map_dtype(dtype)

def substitute_llvm_executors(aterm_graph, executors, operands):
    translator = ATermToAstTranslator(executors, operands)
    return translator.visit(aterm_graph)
