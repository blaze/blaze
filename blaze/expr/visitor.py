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

    def __init__(self):
        self.indent = 0

    def visit(self, tree):

        if tree.children:
            print ('\t'*self.indent) + tree.__class__.__name__
            self.indent += 1
            for node in [self.visit(i) for i in tree.children]:
                print node
            self.indent -= 1
        else:
            return ('\t'*self.indent) + tree.__class__.__name__

    def Unknown(self, tree):
        raise NoVisitor(tree)

#------------------------------------------------------------------------
# Transformers
#------------------------------------------------------------------------

# TODO: write transformer counterparts?

class BasicGraphVisitor(object):
    """
    Visit an aterm graph. Each term must have a handler or Unknown is called.
    """

    # TODO: more robust!
    def visit_node(self, tree):
        nodei = tree.__class__.__name__
        trans = getattr(self, nodei, False)
        if trans:
            return trans(tree)
        else:
            return self._unknown(tree)

    def visit(self, tree):
        if isinstance(tree, list):
            return [self.visit(i) for i in tree]
        else:
            return self.visit_node(tree)

    def _unknown(self, tree):
        return self.Unknown(tree)

    def Unknown(self, tree):
        raise NoVisitor(tree)

class GraphVisitor(BasicGraphVisitor):
    """
    Similar to BasicGraphVisitor, but visits the children of unhandled terms.
    """

    def _unknown(self, tree):
        return self.visitchildren(tree)

    def visitchildren(self, tree):
        for fieldname in tree._fields:
            field = getattr(tree, fieldname)
            self.visit(field)

class GraphTransformer(GraphVisitor):
    """
    Similar to GraphVisitor, but return values replace the terms in the graph.
    """

    def _unknown(self, tree):
        return self.visitchildren(tree)

    def visit(self, tree):
        if isinstance(tree, list):
            result = [self.visit(i) for i in tree]
            return [x for x in result if x is not None]
        else:
            return self.visit_node(tree)

    def visitchildren(self, tree):
        for fieldname in tree._fields:
            field = getattr(tree, fieldname)
            result = self.visit(field)
            setattr(tree, fieldname, result)

        return tree


class MroVisitor(object):

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
