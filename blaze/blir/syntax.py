from _ast import AST

# We recycle the Python AST mostly as a convienance so that we can
# exploit the machinery that exists in the standard library.

class Module(AST):
    _fields = ['body']

    def __repr__(self):
        return "Module(%s)" % (repr(self.body),)

class Const(AST):
    _fields = ['value']

    def __repr__(self):
        return "Const(%s)" % (repr(self.value),)

class Tuple(AST):
    _fields = ['elts']

    def __repr__(self):
        return "Tuple(%s)" % (repr(self.elts),)

class Type(AST):
    _fields = ['name']

    def __repr__(self):
        return "Type(%s)" % (repr(self.name),)

class ParamType(AST):
    _fields = ['cons', 'arg']

    def __repr__(self):
        return "ParamType(%s,%s)" % (repr(self.cons),repr(self.arg))

class Project(AST):
    _fields = ['name', 'field']

    def __repr__(self):
        return "Proj(%s, %s)" % (repr(self.name), repr(self.field))

class LoadVariable(AST):
    _fields = ['name']

    def __repr__(self):
        return "LoadVariable(%s)" % (repr(self.name),)

class StoreVariable(AST):
    _fields = ['name']

    def __repr__(self):
        return "StoreVariable(%s)" % (repr(self.name),)

class LoadIndex(AST):
    _fields = ['name', 'indexer']

    def __repr__(self):
        return "LoadIndex(%s,%s)" % (repr(self.name), repr(self.indexer))

class StoreIndex(AST):
    _fields = ['name', 'indexer']

    def __repr__(self):
        return "StoreIndex(%s,%s)" % (repr(self.name), repr(self.indexer))

class UnaryOp(AST):
    _fields = ['op','expr']

    def __repr__(self):
        return "UnaryOp(%s)" % (repr(self.op), repr(self.expr))

class BinOp(AST):
    _fields = ['op','left','right']

    def __repr__(self):
        return "BinOp(%s,%s,%s)" % (repr(self.op), repr(self.left), repr(self.right))

class FunctionCall(AST):
    _fields = ['name','arglist']

    def __repr__(self):
        return "FunctionCall(%s,%s)" % (repr(self.name), repr(self.arglist))

class Assign(AST):
    _fields = ['store_location','expr']

    def __repr__(self):
        return "Assign(%s,%s)" % (repr(self.store_location), repr(self.expr))

class Print(AST):
    _fields = ['expr']

    def __repr__(self):
        return "Print(%s)" % (repr(self.expr),)

class Range(AST):
    _fields = ['start', 'stop']

    def __repr__(self):
        return "Range(%s, %s)" % (repr(self.start),repr(self.stop))

class Statements(AST):
    _fields = ['statements']

    def __repr__(self):
        return "Statements(%s)" % (repr(self.statements),)

class VarDecl(AST):
    _fields = ['typename','name', 'expr']

    def __repr__(self):
        return "VarDecl(%s,%s,%s)" % (repr(self.typename),repr(self.name),repr(self.expr))

class ConstDecl(AST):
    _fields = ['name','expr']

    def __repr__(self):
        return "ConstDecl(%s,%s)" % (repr(self.name),repr(self.expr))

class ParmDeclaration(AST):
    _fields = ['name', 'typename']

    def __repr__(self):
        return "ParmDeclaration(%s,%s)" % (repr(self.name),repr(self.typename))

class FunctionSig(AST):
    _fields = ['name', 'parameters', 'typename']

    def __repr__(self):
        return "FunctionSig(%s,%s,%s)" % (repr(self.name),repr(self.parameters), repr(self.typename))

class ExternFuncDecl(AST):
    _fields = ['cconv', 'sig']

    def __repr__(self):
        return "ExternFuncDecl(%s,%s)" % (repr(self.cconv), repr(self.sig))

class Compare(AST):
    _fields = ['op', 'left', 'right']

    def __repr__(self):
        return "Compare(%s,%s,%s)" % (repr(self.op), repr(self.left), repr(self.right))

class IfElseStatement(AST):
    _fields = ['condition', 'if_statements','else_statements']

    def __repr__(self):
        return "IfElseStatement(%s,%s,%s)" % (repr(self.condition), repr(self.if_statements), repr(self.else_statements))

class WhileStatement(AST):
    _fields = ['condition', 'statements']

    def __repr__(self):
        return "WhileStatement(%s,%s)" % (repr(self.condition), repr(self.statements))

class ForStatement(AST):
    _fields = ['var', 'iter', 'body']

    def __repr__(self):
        return "ForStatement(%s, %s, %s)" % (repr(self.var), repr(self.iter), repr(self.body))

class ReturnStatement(AST):
    _fields = ['expr']

    def __repr__(self):
        return "ReturnStatement(%s)" % (repr(self.expr),)

class FunctionDef(AST):
    _fields = ['sig','statements']

    def __repr__(self):
        return "FunctionDef(%s,%s)" % (repr(self.sig),repr(self.statements))
