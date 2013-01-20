"""
Graph visitors.
"""

class NoVisitor(Exception):
    def __init__(self, *args):
        self.args = args
    def __str__(self):
        return 'No transformer for Node: %s' % repr(self.args)

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

class GraphTranslator(GraphTransformer):
    """
    Similar to GraphTransformer, but it allows easy translation of the graph
    to other forms of graph, while mutating the graph at the same time.

    Example:

        We want to translate a portion of our graph to a Python AST, and
        replace the translated part of the graph with a single graph node
        reflecting this transformation.

            1) In each translatable node, set 'self.result = PythonAstNode()'.
               Read results of children by reading 'self.result' after visiting
               the respective children.

            2) For every translatable non-root node, return None to delete
               the node.

            3) At the root of the translatable sub-graph, return the node we
               want in our graph.
    """

    def set_resultlist(self, results):
        if len(results) == 0:
            results = None
        elif len(results) == 1:
            results = results[0]

        self.result = results

    def visit(self, tree):
        self.result = None
        if isinstance(tree, list):
            children = []
            results = []
            for child in tree:
                child = self.visit(child)
                if child is not None:
                    children.append(child)
                if self.result is not None:
                    results.append(self.result)

            self.set_resultlist(results)
            return [c for c in children if c is not None]
        else:
            return self.visit_node(tree)

    def visitchildren(self, tree):
        results = []
        for fieldname in tree._fields:
            field = getattr(tree, fieldname)
            result = self.visit(field)
            setattr(tree, fieldname, result)

            if self.result is not None:
                results.append(self.result)

        self.set_resultlist(results)
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
