#------------------------------------------------------------------------
# Graph Objects
#------------------------------------------------------------------------

class Node(object):
    """ Represents a node in the expression graph which Blaze compiles into
    a program for execution engine.
    """
    _fields = ['children']

    def __init__(self, children):
        self.children = children

    def eval(self):
        """ Evaluates the expression graph """
        # setup a default pipeline
        from blaze.compile import _compile
        ctx, plan = _compile(self)

        # submit to the runtime for the result
        #return execplan(ctx, plan)
        return ctx, plan

    def __iter__(self):
        """ Walk the graph, left to right """
        for a in self.children:
            if isinstance(a, Node):
                yield a
                for b in iter(a):
                    yield b
            elif a is None:
                pass
            else:
                raise TypeError('Invalid children')

    def hash(self):
        # tree hashing, xor with the hash of children
        h = hash(type(self))
        for child in self.children:
            h ^= hash(child)
        return h

    @property
    def name(self):
        raise NotImplementedError
        return 'GenericNode'
