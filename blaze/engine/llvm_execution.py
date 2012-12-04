import ast

from numba import decorators
from numba.ufunc_builder import UFuncBuilder

from blaze.expr import visitor
from blaze.expr import ops
from blaze.engine import pipeline
from blaze.expr import paterm

from blaze.datashape import datashape
from blaze import Table, NDTable, Array, NDArray


class GraphToAst(visitor.ExprTransformer):
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

class ATermToAst(visitor.ExprTransformer):
    """
    Convert an aterm graph to a Python AST.
    """

    opname_to_astop = {
        'add': ast.Add,
        'mul': ast.Mult,
    }

    def __init__(self):
        super(ATermToAst, self).__init__()
        self.ufunc_builder = UFuncBuilder()

    def AAppl(self, app):
        if paterm.matches('Arithmetic;*', app.spine):
            opname = app.args[0].lower()
            op = self.opname_to_astop.get(opname, None)
            args = app.args[1:]
            if op is not None:
                if len(args) == 2:
                    left, right = self.visit(args)
                    return ast.BinOp(left=left, op=op(), right=right)

        return self.Unknown(app)

    def Unknown(self, tree):
        return self.ufunc_builder.register_operand(tree)


def getsource(ast):
    from meta import asttools
    return asttools.dump_python_source(ast).strip()


def convert_aterm(aterm_graph):
    """
    Convert an aterm graph to a Python AST.
    """
    visitor = ATermToAst()
    pyast = visitor.visit(aterm_graph)
    operands = visitor.ufunc_builder.operands
    pyast_function = visitor.ufunc_builder.build_ufunc_ast(pyast)
    return operands, pyast_function


def convert_graph(lazy_blaze_graph):
    """
    >>> a = NDArray([1, 2, 3, 4], datashape('2, 2, int'))
    >>> operands, ast_func = convert_graph(a + a)

    >>> print getsource(ast_func)
    def ufunc0(op0, op1):
        return (op0 + op1)
    >>> print operands
    [Array(4346842064){True}, Array(4346842064){True}]
    """
    # Convert blaze graph to ATerm graph
    p = pipeline.Pipeline()
    result = p.run_pipeline(lazy_blaze_graph)
    # Convert to ast
    return convert_aterm(result)

if __name__ == '__main__':
#    a = NDArray([1, 2, 3, 4], datashape('2, 2, int'))
#    p = pipeline.Pipeline()
#    result = p.run_pipeline(a + a)
#    ops, funcdef = convert_graph(result)
#    print getsource(funcdef)
#    print result

    import doctest
    doctest.testmod()
