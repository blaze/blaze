import ast

from numba import decorators
from numba.ufunc_builder import UFuncBuilder

from ndtable.expr import visitor
from ndtable.expr import ops

from ndtable.datashape import datashape
from ndtable import Table, NDTable, Array, NDArray

binop_to_astop = {
    ops.Add: ast.Add,
    ops.Mul: ast.Mult,
}

class GraphToAst(visitor.ExprTransformer):

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

def getsource(ast):
    from meta import asttools
    return asttools.dump_python_source(ast).strip()

def convert_graph(lazy_blaze_graph):
    """
    >>> a = NDArray([1, 2, 3, 4], datashape('2, 2, int'))
    >>> operands, ast_func = convert_graph(a + a)

    >>> print getsource(ast_func)
    def ufunc0(op0, op1):
        return (op0 + op1)
    >>> print map(type, operands)
    [<class 'ndtable.table.NDArray'>, <class 'ndtable.table.NDArray'>]
    """
    visitor = GraphToAst()
    pyast = visitor.visit(lazy_blaze_graph)

    operands = visitor.ufunc_builder.operands
    pyast_function = visitor.ufunc_builder.build_ufunc_ast(pyast)
    return operands, pyast_function

if __name__ == '__main__':
    import doctest
    doctest.testmod()
