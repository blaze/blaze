from __future__ import absolute_import

"""
A typed expression graph prototype. maintains a distinction between
values and operations, intermediate results are attached to a AST graph
that is threaded through the compositions unbeknownst to the user.

The advantage of a typed graph vs a untyped graph is that we maintain a
API that reflects the "proxied values" so that help() still works and
scalar methods/operations only appear on scalar types and vector methods
only appear on vector types etc.

Many things are not included here:

    - Broadcasting
    - Implicit coercion
    - Type signatures

"""

import imp
import sys
import random

import numpy as np

from ast import AST
from astutils import dump

#------------------------------------------------------------------------
# Kernel Enum
#------------------------------------------------------------------------

MAP     = 0
ZIPWITH = 1
SCAN    = 2
REDUCE  = 3
OUTER   = 4

#------------------------------------------------------------------------
# Module
#------------------------------------------------------------------------

# The only real idea of the "module system" is that we seperate
# implementation of the graph from the specification of operations over
# the graph. If we want to add additional structure to the graph we
# extend the dictionary of types and functions instead of having to
# subclassing the Blaze internals.


# Mapping of types to namespaces
bound_ns = {
    'array': {
        '__add__'  : lambda a, b: kernel(ZIPWITH, 'add', [a, b]),
    },

    'scalar': {
        '__add__'  : lambda a, b: primop('add', [a, b]),
        '__mul__'  : lambda a, b: primop('mul', [a, b]),
    }
}

# Mapping of functions to lookup tables over argument types
anon_ns = {
    'basic': {

        'abs' : {
            ('array',)  : lambda a: kernel(ZIPWITH, 'abs', [a]),
            ('scalar',) : lambda a: primop('abs', [a]),
        },

        'neg' : {
            ('array',)  : lambda a: kernel(MAP, 'neg', [a]),
            ('scalar',) : lambda a: primop('neg', [a]),
        },

        'add' : {
            ('array' , 'array')  : lambda a,b: kernel(ZIPWITH, 'add', [a, b]),
            ('scalar', 'scalar') : lambda a,b: primop('add', a, b),
        },

        'sin' : {
            ('array' ,) : lambda a: kernel(MAP, 'sin', [a]),
            ('scalar',) : lambda a: primop('sin', [a]),
        },

        'cos' : {
            ('array' ,) : lambda a: kernel(MAP, 'cos', [a]),
            ('scalar',) : lambda a: primop('cos', [a]),
        },

        'dot' : {
            ('array', 'array') : lambda a,b: kernel(ZIPWITH, 'dot', [a, b]),
        }

    }
}

def build_module(name, ns):
    mod = imp.new_module('blaze.' + name)
    for fname, table in ns.iteritems():
        setattr(mod, fname, match(fname, table))
    sys.modules['blaze.' + name] = mod
    return mod

#------------------------------------------------------------------------
# Docstrings
#------------------------------------------------------------------------

docs = {
    'array'  : 'A vector value',
    'scalar' : 'A scalar value',
}

#------------------------------------------------------------------------
# Namespaces
#------------------------------------------------------------------------

def unify(op, e1, e2):
    """ Logic for determining return type from operand types"""
    # passthrough for now...
    return e1.ty

def unit(val):
    """ Create expression from value """

    if isinstance(val, Value):
        return val._ast
    elif isinstance(val, Terminal):
        return val
    elif isinstance(val, (int, long, float)):
        return Terminal('scalar', val)
    elif isinstance(val, (list, np.ndarray)):
        return Terminal('array', val)

    elif isinstance(val, object):
        return Terminal('array', val)

    else:
        raise NotImplementedError

def primop(op, args):
    """ Append a primitive operation to the expression tree """
    operands = map(unit, args)
    retty = unify(op, args[0], args[1])
    logic = Op(op, operands)

    return Value(retty, logic)

def kernel(kind, name, args):
    """ Append a kernel operation to the expression tree """
    operands = map(unit, args)
    retty = unify(name, args[0], args[1])
    logic = Kernel(kind, name, operands)

    return Value(retty, logic)

#------------------------------------------------------------------------
# Matching
#------------------------------------------------------------------------

def match(fname, table):
    # need error handling and arity-check here, but for now just ignore
    def matcher(*args):
        sig = tuple([a.ty for a in args])
        try:
            return table[sig](*args)
        except KeyError:
            raise Exception("No matching implementation of '%s' for signature '%s'" % (fname, sig))
    return matcher

def oracle(expr):
    return random.choice(['numpy', 'blir', 'numexpr'])

# This is normally called at modoule registration time, for
# instruction purposese just here...
build_module('basic', anon_ns['basic'])

#------------------------------------------------------------------------
# Proxy Nodes
#------------------------------------------------------------------------

class Node(object):

    @classmethod
    def _proxy(cls, ty):
        global module
        ns = {}

        for name, obj in bound_ns[ty].iteritems():
            ns[name] = obj

        ns['ty'] = ty
        ns['__doc__'] = docs[ty]
        return type("%s" % (cls.__name__), (cls, AST), ns)

    def ast(self):
        return self._ast

    def eval(self, engine=None):
        import compile
        expr = self._ast

        if engine == 'numpy':
            return compile.eval_numpy(expr)
        elif engine == 'blir':
            return compile.eval_blir(expr)
        elif engine == 'numexpr':
            return compile.eval_numexpr(expr)
        else:
            return self.eval(oracle(expr))

#------------------------------------------------------------------------
# Operations
#------------------------------------------------------------------------

class Kernel(AST):
    """ Kernels are functions over containers """
    _fields = ['op', 'name', 'args']

    def __init__(self, kind, name, args):
        self.kind = kind
        self.name = name
        self.args = args

class Op(AST):
    _fields = ['op', 'left', 'right']

    def __init__(self, op, operands):
        self.op = op
        self.operands = operands

#------------------------------------------------------------------------
# Values
#------------------------------------------------------------------------

# The difference between a Value and Terminal is that a Terminal has a
# ``src`` param pointing at the concrete source of the data. A value has
# a ``ast`` param pointing at the ast needed to compute it's value.

class Value(Node):
    """ Values are intermediate computations """
    _fields = []

    def __new__(cls, ty, ast):
        prototype = cls._proxy(ty)
        ins = object.__new__(prototype)
        prototype.__init__(ins, ty, ast)
        return ins

    def __init__(self, ty, ast):
        self.ty = ty
        self._ast = ast

    def __str__(self):
        return '<Value(ty=%s)>' % self.ty

class Terminal(Node):
    """ Terminals are concrete values """
    _fields = ['ty']

    def __new__(cls, ty, src):
        prototype = cls._proxy(ty)
        ins = object.__new__(prototype)
        prototype.__init__(ins, ty, src)
        return ins

    def __init__(self, ty, src):
        self.ty = ty
        self.src = src

    def eval(self):
        return self

    def __str__(self):
        return '<Terminal(src=%s)>' % id(self.src)

#------------------------------------------------------------------------

if __name__ == '__main__':
    import blaze
    from blaze.basic import dot

    T = Terminal

    # ---------------------------------
    a0 = 2
    a1 = 7

    a2 = np.array([3,1,4], dtype='int32')
    a3 = np.array([1,5,9], dtype='int32')
    # ---------------------------------
    t = T('scalar', a0)
    s = T('scalar', a1)

    a = T('array', a2)
    b = T('array', a3)
    # ---------------------------------

    print('A'.center(80, '='))
    print(a+s)
    print(dump((a+b).ast()))

    print('B'.center(80, '='))
    print(t+s)
    print(dump((t+s).ast()))

    print('C'.center(80, '='))
    print(dump(dot(a,a).ast()))

    print('D'.center(80, '='))
    print(dump((a+a+a+b).ast()))

    print((a+b*b).eval(engine='blir'))
    print((a+b).eval(engine='numexpr'))

    print(a+1)
