import itertools

from gen import Gen

c_types = {
    int: 'int',
    float: 'float',
}

#------------------------------------------------------------------------
# Printing Utils
#------------------------------------------------------------------------

def init(xs):
    for x in xs:
        return x

def tail(xs):
    for x in reversed(xs):
        return x

def nln(xs):
    return "\n".join(l.rstrip() for l in xs)

def cat(xs):
    return "".join(xs)

def pgen(self):
    return

#------------------------------------------------------------------------
# C Syntax
#------------------------------------------------------------------------

# Trying to make this close to the ATerm C representation.

class FunctionDef(object):
    pass

class Assign(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def gen(self):
        yield "%s = %s;" % (self.lhs, self.rhs)


class Typedef(object):
    def __init__(self, ty, val, name, struct=False):
        self.ty = ty
        self.val = val
        self.name = name
        self.struct = False

    def gen(self):
        if self.struct:
            raise NotImplementedError
        else:
            yield "typdef %s = %s;" % (self.lhs, self.rhs)

class Decl(object):

    def __init__(self, ty, var, val):
        self.ty = ty
        self.var = var
        self.val = val

    def gen(self):
        if self.val:
            yield "%s %s = %s;" % (c_types[self.ty], self.var, self.val)
        else:
            yield "%s %s;" % (c_types[self.ty], self.var)

class For(object):
    def __init__(self, start, cond, itr, body):
        self.start = start
        self.cond = cond
        self.itr = itr
        self.body = body

    def gen(self):
        yield "for (%s; %s; %s)" % (self.start, self.cond, self.itr)
        yield list(self.body)

class Block(object):

    def __init__(self, inner=None):
        self.inner = list(inner or [])

    def gen(self):
        yield "{"
        for stmt in self.inner:
            for st in stmt.gen():
                yield st
        yield "}"

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
        inner = None

        for d in self.dimensions:
            yield Decl(d.name, int, 0)

        for l in reversed(self.dimensions):
            if inner is None:
                yield self.body
            else:
                #yield For()
                inner = d
