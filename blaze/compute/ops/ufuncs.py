# -*- coding: utf-8 -*-

"""
Blaze element-wise ufuncs.
"""

from __future__ import print_function, division, absolute_import
try:
    import __builtin__ as builtins
except ImportError:
    import builtins

from ..function import jit_elementwise

@jit_elementwise('a -> a -> a')
def add(a, b):
    return a + b

@jit_elementwise('a -> a -> a')
def mul(a, b):
    return a * b

@jit_elementwise('a -> a -> a')
def sub(a, b):
    return a - b

@jit_elementwise('a -> a -> a')
def div(a, b):
    return a / b

@jit_elementwise('a -> a -> a')
def truediv(a, b):
    return a / b

@jit_elementwise('a -> a -> a')
def floordiv(a, b):
    return a // b

@jit_elementwise('a -> a -> a')
def mod(a, b):
    return a % b

@jit_elementwise('a -> a')
def neg(a):
    return -a

#------------------------------------------------------------------------
# Compare
#------------------------------------------------------------------------

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def eq(a, b):
    return a == b

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def ne(a, b):
    return a != b

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def lt(a, b):
    return a < b

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def le(a, b):
    return a <= b

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def gt(a, b):
    return a > b

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def ge(a, b):
    return a >= b

#------------------------------------------------------------------------
# Logical
#------------------------------------------------------------------------

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def logical_and(a, b):
    return a and b

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def logical_or(a, b):
    return a or b

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def logical_xor(a, b):
    return bool(a) ^ bool(b)

@jit_elementwise('A..., T -> A..., bool')
def logical_not(a):
    return not a

#------------------------------------------------------------------------
# Bitwise
#------------------------------------------------------------------------

@jit_elementwise('A..., T : integral -> A..., T -> A..., T')
def bitwise_and(a, b):
    return a & b

@jit_elementwise('A..., T : integral -> A..., T -> A..., T')
def bitwise_or(a, b):
    return a | b

@jit_elementwise('A..., T : integral -> A..., T -> A..., T')
def bitwise_xor(a, b):
    return a ^ b

@jit_elementwise('A..., T : integral -> A..., T -> A..., T')
def left_shift(a, b):
    return a << b

@jit_elementwise('A..., T : integral -> A..., T -> A..., T')
def right_shift(a, b):
    return a >> b

# ______________________________________________________________________

# Aliases

subtract        = sub
multiply        = mul
true_divide     = truediv
floor_divide    = floordiv
equal           = eq
not_equal       = ne
less            = lt
less_equal      = le
greater         = gt
greater_equal   = ge

#------------------------------------------------------------------------
# Math
#------------------------------------------------------------------------

@jit_elementwise('A -> A')
def abs(x):
    return builtins.abs(x)
