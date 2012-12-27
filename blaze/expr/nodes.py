#------------------------------------------------------------------------
# Graph Objects
#------------------------------------------------------------------------

class Node(object):
    """ Represents a node in the expression graph which Blaze compiles into
    a program for execution engine.
    """
    # Use __slots__ so we don't incur the full cost of a class
    __slots__ = ['children']
    _fields = ['children']

    def __init__(self, children):
        self.children = children

    def eval(self):
        """ Evaluates the expression graph """
        from blaze.engine import pipeline
        from blaze.rts.execution import execplan

        # setup a default pipeline
        line = pipeline.Pipeline()

        # generate the plan
        ctx, plan = line.run_pipeline(self)

        # submit to the runtime for the result
        return execplan(ctx, plan)

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

    def __coiter__(self, co):
        """
        Walk the graph, left to right with a coroutine.
        """
        children = dict(enumerate(self)).viewitems()

        def switch(child):
            changed = co.send(child)
            if changed:
                children[idx] = changed
            else:
                del children[idx]

        for idx, child in children:

            if isinstance(child, Node):
                switch(child)

            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, Node):
                        switch(child)

    #def __hash__(self):
        ## tree hashing, xor with the hash of children
        #h = hash(type(self))
        #for child in self.children:
            #h ^= hash(child)
        #return h

    @property
    def name(self):
        return 'GenericNode'
