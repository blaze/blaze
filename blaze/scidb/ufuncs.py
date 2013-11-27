# -*- coding: utf-8 -*-

"""
SciDB implementations element-wise ufuncs.
"""

from __future__ import print_function, division, absolute_import

from blaze.function import function
from blaze.datashape import typesets, datetime64
from blaze.ops import ufuncs
from blaze.ops.ufuncs import (add, mul, sub, div, truediv, floordiv, mod,
                              eq, ne, ge, gt, le, lt,
                              logical_and, logical_or, logical_not, logical_xor,
                              bitwise_and, bitwise_or, bitwise_xor)
from .kernel import scidb_elementwise
from .query import apply, expr, iff, qformat

real = typesets.integral | typesets.floating

#------------------------------------------------------------------------
# Implement functions
#------------------------------------------------------------------------

def define_unop(signature, name, op):
    """Define a unary scidb operator"""
    def unop(op, x):
        return qformat("({0} {1})", op, x)

    unop.__name__ = name
    _implement(unop, signature)
    return unop

def define_binop(signature, name, op):
    """Define a binary scidb operator"""
    def binop(op, a, b):
        return qformat("({0} {1} {2})", a, op, b)

    binop.__name__ = name
    _implement(binop, signature)
    return binop

def _implement(f, signature):
    name = f.__name__
    prevop = getattr(ufuncs, name)
    prevop.implement_by_sig(signature, 'scidb', f)

#------------------------------------------------------------------------
# Arithmetic
#------------------------------------------------------------------------

add = define_binop("a : real -> a -> a", "add", "+")
mul = define_binop("a : real -> a -> a", "add", "*")
sub = define_binop("a : real -> a -> a", "add", "-")
div = define_binop("a : real -> a -> a", "add", "/")
# floordiv = define_binop("a : real -> a -> a", "add", "//")
# truediv = define_binop("a : real -> a -> a", "add", "/")
mod = define_binop("a : real -> a -> a", "add", "%")

neg = define_unop("a -> a", "neg", "-")

#------------------------------------------------------------------------
# Compare
#------------------------------------------------------------------------

eq = define_binop("a..., T -> a..., T -> a..., bool", "add", "==")
ne = define_binop("a..., T -> a..., T -> a..., bool", "add", "!=")
lt = define_binop("a..., T -> a..., T -> a..., bool", "add", "<")
le = define_binop("a..., T -> a..., T -> a..., bool", "add", "<=")
gt = define_binop("a..., T -> a..., T -> a..., bool", "add", ">")
ge = define_binop("a..., T -> a..., T -> a..., bool", "add", ">=")

#------------------------------------------------------------------------
# Logical
#------------------------------------------------------------------------

# TODO: We have to implement all combinations of types here for 'and' etc,
#       given the set {numeric, bool} for both arguments. Overloading at the
#       kernel level would reduce this. Can we decide between "kernels" and
#       "functions" depending on the inference process.

# TODO: numeric/bool and bool/numeric combinations

# --- and ---

@scidb_elementwise('A..., T : numeric -> A..., T -> A..., bool')
def logical_and(a, b):
    return iff(ibool(a). ibool(b), false)

@scidb_elementwise('A..., bool -> A..., bool -> A..., bool')
def logical_and(a, b):
    return iff(a, b, false)

# --- or ---

@scidb_elementwise('A..., T : numeric -> A..., T -> A..., bool')
def logical_or(a, b):
    return iff(ibool(a), true, ibool(b))

@scidb_elementwise('A..., bool -> A..., bool -> A..., bool')
def logical_or(a, b):
    return iff(a, true, b)

# --- xor ---

# Fixme: repeat of subexpression leads to exponential code generation !

@scidb_elementwise('A..., T : numeric -> A..., T -> A..., bool')
def logical_xor(a, b):
    return iff(ibool(a), logical_not(ibool(b)), ibool(b))

@scidb_elementwise('A..., bool -> A..., bool -> A..., bool')
def logical_xor(a, b):
    return iff(a, logical_not(b), b)

# --- not ---

@scidb_elementwise('A..., T -> A..., bool')
def logical_not(a):
    return apply("not", a)

#------------------------------------------------------------------------
# Math
#------------------------------------------------------------------------

@scidb_elementwise('A : numeric -> A')
def abs(x):
    # Fixme: again exponential codegen
    return iff(lt(x, 0), neg(x), x)

#------------------------------------------------------------------------
# Helper functions
#------------------------------------------------------------------------

def ibool(x):
    return ne(x, "0")

#------------------------------------------------------------------------
# Data types
#------------------------------------------------------------------------

true  = "true"
false = "false"