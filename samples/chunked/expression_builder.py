""" Code to build and visit expressions

In a quite simple way
"""

class _ASTNode(object):
    # ------------------------------------------------------------
    # operators - used to build an AST of the expresion
    def __add__(self, rhs):
        return Operation('+', self, rhs)

    def __sub__(self, rhs):
        return Operation('-', self, rhs)

    def __mul__(self, rhs):
        return Operation('*', self, rhs)

    def dot(self, rhs):
        return Operation('dot', self, rhs)


class Operation(_ASTNode):
    """ simple binary operator -enough for our samples- """

    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return ('Operation(' + repr(self.op) + ', '
                + repr(self.lhs) + ', '
                + repr(self.rhs) + ')')


class Terminal(_ASTNode):
    """ a leaf object containing a source operand. """
    def __init__(self, src):
        self.source = src

    def __repr__(self):
        return 'Terminal(' + repr(self.source) + ')'


class Visitor(object):
    """ a simple visitor for _ASTNodes """
    _method_map = { Operation: 'accept_operation',
                    Terminal: 'accept_terminal' }

    def accept(self, node):
        try:
            _method = getattr(self, 
                              self._method_map[node.__class__])
        except AttributeError:
            _method = getattr(self, 'accept_default')
        except KeyError:
            raise Exception('Unknown class for visitor')

        return _method(node)

    def accept_default(self):
        pass


class PrintVisitor(Visitor):
    def accept_default(self, node):
        print node

    def accept_operation(self, node):
        print 'op ' + node.op
        self.accept(node.lhs)
        self.accept(node.rhs)



if __name__ == '__main__':
    T = Terminal
    expr = T('a') + T('b')*T('c')
    print expr
    PrintVisitor().accept(expr)
