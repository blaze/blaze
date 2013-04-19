"""
Context free syntatic generators for BLIR kernels.
"""

from combinators import *
from textwrap import dedent as dd

#------------------------------------------------------------------------
# Syntax
#------------------------------------------------------------------------

class FuncDef(object):
    """ def {name} ({args}) -> {ret} {{ \n {body} \n }} """

    def __init__(self, name, args, ret, body):
        self.name = name
        self.args = args
        self.ret = ret
        self.body = body

    def __str__(self):
        ctx = dict(
            name = self.name,
            args = ', '.join(str(arg) for arg in self.args),
            ret  = self.ret,
            body = iblock(4, str(self.body)),
        )
        return dd(self.__doc__).format(**ctx)

class Arg(object):
    """%s : %s"""

    def __init__(self, ty, name):
        self.ty = ty
        self.name = name

    def __str__(self):
        return dd(self.__doc__) % (self.name, self.ty)

class VarDecl(object):
    """ var %s %s = %s; """

    def __init__(self, ty, name, val, struct=False):
        self.ty = ty
        self.val = val
        self.name = name

    def __str__(self):
        if self.val == None:
            return "var %s %s" % (self.ty, self.name)
        else:
            return dd(self.__doc__) % (self.ty, self.name, self.val)

class Range(object):
    """ range(%s, %s) """
    def __init__(self, inf, sup):
        self.inf = inf
        self.sup = sup

    def __str__(self):
        return dd(self.__doc__) % (self.inf, self.sup)

class Assign(object):
    """ %s = %s ; """

    def __init__(self, target, value):
        self.target = target
        self.value = value

    def __str__(self):
        return dd(self.__doc__) % (self.target, self.value)

class For(object):
    """ for {var} in {iter} {{ \n {body} \n }} """

    def __init__(self, var, iter, body):
        self.var = var
        self.iter = iter
        self.body = body

    def __str__(self):
        ctx = dict(
            var  = self.var,
            iter = self.iter,
            body = iblock(4, str(self.body)),
        )
        return dd(self.__doc__).format(**ctx)

class Return(object):
    """ return %s; """

    def __init__(self, val):
        self.val = val

    def __str__(self):
        return dd(self.__doc__) % self.val

class Block(object):

    def __init__(self, contents):
        self.contents = contents

    def __str__(self):
        return vcat(str(stmt) for stmt in self.contents)
