"""
This module does most of the standard C type checking and returns
Python-esque error messages for common errors.
"""

import btypes
import intrinsics
from syntax import *
from errors import error

from ast import AST, NodeVisitor
from collections import defaultdict

#------------------------------------------------------------------------
# Symbol Table
#------------------------------------------------------------------------

GLOBAL = 0

class SymbolTable(object):

    def __init__(self):
        self._locals = defaultdict(dict)
        self._globals = {}

    def is_global(self, symbol):
        return symbol in self._globals

    def is_local(self, symbol):
        return any(symbol in scope for scope in self._locals)

    def extend(self, scope, name, node):
        if scope == GLOBAL:
            self._globals[name] = node
        else:
            if name in self._locals[scope]:
                otherdef = self._locals[scope][name]

                # If we're dealing with an AST propogate error
                if isinstance(node, AST):
                    error(node.lineno,
                        "%s already defined. Previous definition on line %s"
                        % (name, getattr(otherdef, "lineno", "<unknown>"))
                    )
            else:
                self._locals[scope][name] = node

    def visible(self, symbol, scope):
        return self.lookup(symbol, scope) is not None

    def lookup(self, symbol, scope=None):
        if scope == GLOBAL:
            return self._globals.get(symbol)
        else:
            return self._globals.get(symbol) or self._locals[scope].get(symbol)

#------------------------------------------------------------------------
# Type Checker
#------------------------------------------------------------------------

class TypeChecker(NodeVisitor):

    def __init__(self, typesystem):
        self.symtab = SymbolTable()
        self.can_declare = True
        self.has_return = False
        self.typesystem = typesystem

        # mutated by visits
        self._current_scope = GLOBAL

        # references to type names
        for ty in typesystem.__all__:
            typ = getattr(typesystem, ty)
            self.symtab.extend(GLOBAL, typ.name, typ)
            #ty.is_global = True

    @property
    def scope(self):
        if self._current_scope == GLOBAL:
            return GLOBAL
        else:
            return self._current_scope.sig.name

    def visit_Module(self, node):
        self.visit(node.body)

    #------------------------------------------------------------------------

    def visit_Const(self, node):
        if isinstance(node.value, bool):
            node.type = btypes.bool_type
        elif isinstance(node.value, int):
            node.type = btypes.int_type
        elif isinstance(node.value, float):
            node.type = btypes.float_type
        elif isinstance(node.value, str):
            node.type = btypes.string_type
        else:
            error(node.lineno,"Value Error: Unknown constant type %s" % type(node.value))
            node.type = btypes.undefined

    def visit_UnaryOp(self, node):
        self.visit(node.expr)

        if node.op not in node.expr.type.unary_ops:
            error(node.lineno,"Unsupported unary operator %s for type %s" % (node.op, node.expr.type.name))
        node.type = node.expr.type

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)

        if node.left.type != node.right.type:
            error(node.lineno,"Type Error: Mismatched operands for %s: '%s' '%s'" % (node.op, node.left.type.name, node.right.type.name))
            node.type = btypes.undefined
        elif node.op not in node.left.type.binary_ops:
            error(node.lineno,"Type Error: Unsupported operator %s: for type '%s'" % (node.op, node.left.type.name))
            node.type = btypes.undefined
        else:
            node.type = node.left.type

    def visit_FunctionCall(self, node):
        symnode = self.symtab.lookup(node.name, self.scope)

        # call to an intrinsics
        if node.name in intrinsics.llvm_intrinsics:
            for n in node.arglist:
                self.visit(n)

            # just pop the last type off the signature
            retty = getattr(intrinsics, node.name)[2][-1]
            node.type = retty

        # call to a named function
        elif symnode is not None:

            if not isinstance(symnode, FunctionSig):
                error(node.lineno,"Type Error: '%s' is not callable" % node.name)
                node.type = btypes.undefined
            else:
                if len(node.arglist) != len(symnode.parameters):
                    error(node.lineno,"Type Error: %s expected %d arguments. Got %d." % (node.name, len(symnode.parameters), len(node.arglist)))

                for n, (arg, parm) in enumerate(zip(node.arglist, symnode.parameters),1):
                    self.visit(arg)
                    arg.type == parm.type
                    if arg.type != parm.type:
                        error(arg.lineno,"Type Error. Argument %d must be %s" % (n, parm.type.name))

                node.type = symnode.type

        else:
            error(node.lineno,"Name Error: name %s is not defined" % node.name)
            node.type = btypes.undefined

    def visit_Assign(self, node):
        self.visit(node.store_location)
        self.visit(node.expr)

        # array assignment
        if isinstance(node.store_location.type, btypes.TParam):
            node.store_location.expr = node.expr

        # scalar assignement
        else:
            if node.store_location.type != node.expr.type:
                error(node.lineno,"Type Error: Assignment not possible %s != %s" % (node.store_location.type.name, node.expr.type.name))
            node.store_location.expr = node.expr

    def visit_ConstDecl(self, node):
        # We follow the C-- convention in that constants do not need
        # to be explictly type and just infer from the literal token.
        self.visit(node.expr)

        if not isinstance(node.expr, Const):
            error(node.lineno, "Type Error: '%s' initializer element is not a constant" % node.name)
        else:
            node.type = node.expr.type
            self.symtab.extend(GLOBAL, node.name, node)
            node.is_global = True

    def visit_VarDecl(self, node):
        if not self.can_declare:
            error(node.lineno, "Cannot declare variables in non-toplevel scope.")
            return

        self.visit(node.typename)
        node.type = node.typename.type

        if isinstance(node.type, btypes.TParam):
            error(node.lineno,"Type Error: Cannot allocate arrays.")

        if node.expr:
            self.visit(node.expr)
            if node.expr.type != node.type:
                error(node.lineno,"Type Error %s != %s" % (node.type.name, node.expr.type.name))
        else:
            # If a variable is unitialzied and it has arithmetic type,
            # it is initialized set to it's types zero value
            node.expr = Const(node.type.zero)
            node.expr.type = node.type

        self.symtab.extend(self.scope, node.name, node)
        node.is_global = False

    def visit_ParmDeclaration(self, node):
        self.visit(node.typename)

        if isinstance(node.typename, ParamType):
            ptype = node.typename
            node.type = btypes.TParam(ptype.cons, ptype.arg)
        else:
            node.type = node.typename.type

    def visit_FunctionSig(self, node):
        for param in node.parameters:
            self.visit(param)
        self.visit(node.typename)
        node.type = node.typename.type
        self.symtab.extend(GLOBAL, node.name, node)

    def visit_ParamType(self, node):
        self.visit(node.cons)
        self.visit(node.arg)

        node.type = btypes.TParam(node.cons, node.arg)

    def visit_Type(self, node):
        ty = self.symtab.lookup(node.name, self.scope)
        if ty:
            if isinstance(ty, btypes.Type):
                node.type = ty
            else:
                error(node.lineno, "Syntax Error: '%s' is not a type" % node.name)
                node.type = btypes.undefined
        else:
            error(node.lineno, "Name Error: Type '%s' is not defined" % node.name)
            node.type = btypes.undefined

    #------------------------------------------------------------------------

    def visit_LoadVariable(self, node):
        sym = self.symtab.lookup(node.name, self.scope)
        if sym:
            if isinstance(sym, (ConstDecl, VarDecl, ParmDeclaration)):
                node.type = sym.type
            else:
                error(node.lineno,"Type Error: %s not a valid location" % node.name)
                node.type = btypes.undefined
        else:
            error(node.lineno,"Name Error: '%s' undeclared (first use in this function)" % node.name)
            node.type = btypes.undefined

        # Annotate with the type
        node.sym = sym

    def visit_StoreVariable(self, node):
        sym = self.symtab.lookup(node.name, self.scope)

        if sym:
            if isinstance(sym, (VarDecl, ParmDeclaration)):
                node.type = sym.type
            elif isinstance(sym, ConstDecl):
                error(node.lineno,"Type Error: %s is constant" % node.name)
                node.type = sym.type
            else:
                error(node.lineno,"Type Error: %s not a valid location" % node.name)
                node.type = btypes.undefined
        else:
            error(node.lineno,"Name Error: '%s' undeclared (first use in this function)" % node.name)
            node.type = btypes.undefined

        # Annotate with the type
        node.sym = sym

    def visit_LoadIndex(self, node):
        # XXX: need to assert a lot of things about the index
        # types in the typecheck phase, bounds checking
        sym = self.symtab.lookup(node.name, self.scope)
        self.visit(node.indexer)

        if sym is not None:
            if isinstance(sym.type, btypes.TParam):
                # extract the element type
                # <array[int]> -> int
                node.type = sym.type.arg.type
            else:
                error(node.lineno,"Type Error: Cannot index scalar type: %s" % node.name)
                node.type = btypes.undefined
        else:
            error(node.lineno,"Name Error: %s undefined" % node.name)
            node.type = btypes.undefined

    def visit_StoreIndex(self, node):
        sym = self.symtab.lookup(node.name, self.scope)
        self.visit(node.indexer)

        if sym:
            if isinstance(sym.type, btypes.TParam):
                # extract the element pointer
                node.type = sym.type.arg.type
            else:
                error(node.lineno,"Type Error: Cannot index scalar type: %s" % node.name)
                node.type = btypes.undefined
        else:
            error(node.lineno,"Name Error: %s undefined" % node.name)
            node.type = btypes.undefined

    def visit_Range(self, node):
        self.visit(node.start)
        self.visit(node.stop)

        if node.start.type != btypes.int_type:
            error(node.lineno, "Type Error: Bounds to range statement must be integers")
        if node.stop.type != btypes.int_type:
            error(node.lineno, "Type Error: Bounds to range statement must be integers")

    #------------------------------------------------------------------------

    def visit_Compare(self, node):
        self.visit(node.left)
        self.visit(node.right)

        if node.left.type != node.right.type:
            error(node.lineno, "Type Error %s %s %s" % (
                node.left.type.name,
                node.op,
                node.right.type.name
            ))
        elif node.op not in node.left.type.cmp_ops:
            error(node.lineno, "Unsupported operator %s for type %s" % (node.op, node.left.type.name))

        node.type = btypes.bool_type

    def visit_IfElseStatement(self, node):
        self.can_declare = False

        self.visit(node.condition)
        if node.condition.type != btypes.bool_type:
            error(node.lineno, "Value Conditional expression must evaluate to bool")

        self.visit(node.if_statements)

        if_returns = self.has_return
        self.has_return = False

        if node.else_statements is not None:
            self.visit(node.else_statements)
            else_returns = self.has_return
        else:
            else_returns = False

        self.has_return = if_returns and else_returns
        self.can_declare = True

    def visit_WhileStatement(self, node):
        self.can_declare = False

        if self.scope is GLOBAL:
            error(node.lineno, "Syntax Error: while loop outside of function")
        else:
            self.visit(node.condition)
            if node.condition.type != btypes.bool_type:
                error(node.lineno, "Value Error: Conditional expression must evaluate to bool")
            self.visit(node.statements)

        self.can_declare = True

    def visit_ForStatement(self, node):
        self.can_declare = False

        self.visit(node.iter)
        if self.scope is GLOBAL:
            error(node.lineno, "Syntax Error: for loop outside of function")
        else:
            self.visit(node.body)

        self.can_declare = True

    def visit_ReturnStatement(self, node):
        if self.scope is GLOBAL:
            error(node.lineno, "Syntax Error: return outside function")
        else:
            self.visit(node.expr)
            fn = self._current_scope

            if node.expr.type != fn.sig.type:
                error(node.lineno, "Type Error: Value returned does match function signature  %s != %s" % (
                        node.expr.type.name, fn.sig.type.name))
            self.has_return = True

    #------------------------------------------------------------------------

    def visit_FunctionDef(self, node):
        if self.scope is not GLOBAL:
            error(node.lineno, "Syntax Error: Cannot define functions within function.")
        else:
            self.visit(node.sig)
            self._current_scope = node
            self.has_return = False

            for parm in node.sig.parameters:
                self.symtab.extend(self.scope, parm.name, parm)

            self.visit(node.statements)
            self._current_scope = GLOBAL

            if (not self.has_return) and (not node.sig.type == btypes.void_type):
                error(node.lineno, "Syntax Error: Control reaches end of non-void function '%s'" % node.sig.name)

#------------------------------------------------------------------------
# Toplevel
#------------------------------------------------------------------------

def typecheck(node, typesystem):
    checker = TypeChecker(typesystem)
    checker.visit(node)
    return checker.symtab

#------------------------------------------------------------------------
# --ddump-tc
#------------------------------------------------------------------------

def ddump_tc(source, verbose=False):
    import sys
    import parser
    import errors
    import pprint

    with errors.listen():
        ast = parser.parse(source)
        symtab = typecheck(ast, verbose)

    if errors.occurred():
        sys.stdout.write("Not well-typed!\n")
    else:
        print('Locals'.center(80, '='))
        sys.stdout.write(pprint.pformat(symtab._locals.items()) + '\n')
        print('Globals'.center(80, '='))
        sys.stdout.write(pprint.pformat(symtab._globals) + '\n')
        sys.stdout.write("Well-typed!\n")
