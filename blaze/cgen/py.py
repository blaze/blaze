from gen import Gen

#------------------------------------------------------------------------
# Python Syntax
#------------------------------------------------------------------------

class FunctionDef(object):
    pass

class Assign(object):
    pass

class Decl(object):

    def __init__(self, var, val):
        self.var = var
        self.val = val

    def gen(self):
        yield "%s = %s" % (self.var, self.val)

class For(object):
    pass

class Block(object):
    pass

#------------------------------------------------------------------------
# Kernel Generation
#------------------------------------------------------------------------

class Dimension(object):
    def __init__(self, name, bound=None):
        self.name = name
        self.bound = bound

class Kernel(object):
    def __init__(self, dimensions, body):
        self.dimensions = dimensions
        self.body = body

    def gen(self):
        pass
