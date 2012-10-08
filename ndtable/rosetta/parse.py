"""
Rosetta stones are a declarative way of building translation tables
between multiple Python namespaces. They can be used to map the
constructs of one numeric Python project to another project.

This could be used in Blaze to "glue" together different PyData
projects and make them understand each other's type systems and
interoperate at the ByteProvider level.
"""

import ast
from pprint import pprint
from string import translate
from ndtable.datashape.parse import Translate, Visitor, op_table

expr_translator = Translate()

class Var(object):
    def __init__(self, *names):
        self.names = names

    def __repr__(self):
        if len(self.names) == 1:
            return str(self.names)
        else:
            return '.'.join(self.names)

class Num(object):
    def __init__(self, val):
        self.val = val

    def __repr__(self):
        return str(self.val)

class Apply(object):
    def __init__(self, *args):
        self.args = args

    def __repr__(self):
        return str(self.args)

class Mapping(object):
    def __init__(self, a,b):
        self.dom = a
        self.cod = b

    def astuple(self):
        # TODO: generally return lambdas
        return (self.dom, self.cod)

    def bound(self):
        """
        Variables that appear in the domain and the codomain.
        """
        return set(self.cod.free_vars()) & set(self.dom.free_vars())

    def globals(self):
        """
        Variables that appear in the codomain but not domain.
        """
        return set(self.cod.free_vars()) - set(self.dom.free_vars())

    def phantom(self):
        """
        Variables that appear in the domain but not the codomain.
        """
        return set(self.dom.free_vars()) - set(self.cod.free_vars())

    def __repr__(self):
        return str(self.dom) + " := " + str(self.cod)

class Rosetta(Visitor):

    def Module(self, tree):
        return [self.visit(i) for i in tree.body]

    def Import(self, tree):
        pass

    def Attribute(self, tree):
        stem = tree.value.id
        return Var(stem, tree.attr)

    def Expr(self, tree):
        return self.visit(tree.value)

    def Call(self, tree):
        fn = self.visit(tree.func)
        args = self.visit(tree.args)
        return Apply(fn, args)

    def Name(self, tree):
        stem = tree.id
        py = getattr(self.namespace, stem, None)
        return py or Var(stem)

    def Num(self, tree):
        return Num(tree.n)

    def Compare(self, tree):
        if isinstance(tree.ops[0],ast.Is):
            left = self.visit(tree.left)
            right = expr_translator.visit(tree.comparators[0])
            return Mapping(left, right)

def parse(expr_string):
    expr = translate(expr_string, op_table)
    return ast.parse(expr)
