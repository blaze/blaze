from pycparser import c_ast
from pycparser import parse_file

class FuncDefVisitor(c_ast.NodeVisitor):

    def __init__(self):
        self.decls = {}

    def visit_Decl(self, node):
        if isinstance(node.type, c_ast.FuncDecl):
            self.decls[node.name] = node.type.args

def parse_header(lib, names):
    ast = parse_file(lib, use_cpp=True)

    v = FuncDefVisitor()
    v.visit(ast)
    return [v.decls[name] for name in names]

if __name__ == '__main__':
    lib = 'cblas.h'
    print parse_header(lib, ['cblas_sdot'])
