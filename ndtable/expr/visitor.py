"""
Graph visitors.
"""

class NoVisitor(Exception):
    def __init__(self, args):
        self.args = args
    def __str__(self):
        return 'No transformer for Node: %s' % repr(self.args)

#------------------------------------------------------------------------
# Pretty Printing
#------------------------------------------------------------------------

class ExprPrinter(object):

    # TODO: more robust!
    def __init__(self):
        self.indent = 0

    def visit(self, tree):

        if self.indent == 0:
            print '\n===================='

        if tree.children:
            print ('    '*self.indent) + tree.__class__.__name__
            self.indent += 1
            [self.visit(i) for i in tree.children]
            self.indent -= 1
        else:
            print ('    '*self.indent) + tree.__class__.__name__

    def Unknown(self, tree):
        raise NoVisitor(tree)

class MorphismPrinter(object):

    # TODO: more robust!
    def __init__(self):
        self.indent = 0

    def visit(self, tree):

        if self.indent == 0:
            print '\n===================='

        if tree.children:
            if hasattr(tree, 'cod'):
                print ('    '*self.indent) + tree.__class__.__name__ + " :: " + str(tree.cod)
            else:
                print ('    '*self.indent) + tree.__class__.__name__

            self.indent += 1
            [self.visit(i) for i in tree.children]
            self.indent -= 1
        else:
            if hasattr(tree, 'cod'):
                print ('    '*self.indent) + tree.__class__.__name__ + " :: " + str(tree.cod)
            else:
                print ('    '*self.indent) + tree.__class__.__name__

    def Unknown(self, tree):
        raise NoVisitor(tree)

#------------------------------------------------------------------------
# Transformers
#------------------------------------------------------------------------

class ExprTransformer(object):

    # TODO: more robust!
    def visit(self, tree):
        if isinstance(tree, list):
            return [self.visit(i) for i in tree]
        else:
            nodei = tree.__class__.__name__
            trans = getattr(self,nodei, False)
            if trans:
                return trans(tree)
            else:
                return self.Unknown(tree)

    def Unknown(self, tree):
        raise NoVisitor(tree)

class MroTransformer(object):

    def visit(self, tree):
        if isinstance(tree, list):
            return [self.visit(i) for i in tree]
        else:
            fn = None
            # Will always fall back to object() if defined.
            for o in tree.__class__.mro():
                nodei = o.__name__
                trans = getattr(self, nodei, False)
                if trans:
                    fn = trans
                    break
                else:
                    continue

            if fn:
                return fn(tree)
            else:
                self.Unknown(tree)

    def Unknown(self, tree):
        raise NoVisitor(tree)
