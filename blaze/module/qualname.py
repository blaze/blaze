#------------------------------------------------------------------------
# Names
#------------------------------------------------------------------------

class Namespace(object):
    def __init__(self, names):
        self.names = names

    def show(self):
        return '.'.join(self.names)

class QualName(object):
    def __init__(self, namespace, name):
        assert isinstance(namespace, list)
        self.namespace = name
        self.name = name

    def isprim(self):
        return self.namespace == ['Prims']

    def isqual(self):
        return len(self.namespace) > 1

    def show(self):
        return '.'.join(self.namespace + [self.name])

    def __str__(self):
        return self.show()

#------------------------------------------------------------------------
# Module
#------------------------------------------------------------------------

class Module(object):

    def __init__(self, name):
        self.name = name

    def alias(self):
        pass

    def expose(self, sym, sig):
        pass

#------------------------------------------------------------------------
# Function References
#------------------------------------------------------------------------

# string -> name
# Reference to a function name
def name(s):
    pass

# name -> term
# Reference to a function name
def ref(n):
    pass

# string -> term
def fn(s):
    pass
