# -*- coding: utf-8 -*-

"""
Blaze element-wise ufuncs.
"""

from __future__ import print_function, division, absolute_import

from blaze.function import overload, elementwise

@elementwise # A -> A -> A
def add(a, b):
    return a + b

@elementwise
def mul(a, b):
    return a * b

@elementwise
def sub(a, b):
    return a - b

@elementwise
def div(a, b):
    return a / b

@elementwise
def truediv(a, b):
    return a / b

@elementwise
def floordiv(a, b):
    return a // b

@elementwise
def mod(a, b):
    return a % b

#------------------------------------------------------------------------
# Compare
#------------------------------------------------------------------------

@elementwise('A -> A -> bool')
def eq(a, b):
    return a == b

@elementwise('A -> A -> bool')
def ne(a, b):
    return a != b

@elementwise('A -> A -> bool')
def lt(a, b):
    return a < b

@elementwise('A -> A -> bool')
def le(a, b):
    return a <= b

@elementwise('A -> A -> bool')
def gt(a, b):
    return a > b

@elementwise('A -> A -> bool')
def ge(a, b):
    return a >= b

#------------------------------------------------------------------------
# Logical
#------------------------------------------------------------------------

@elementwise('A -> A -> bool')
def logical_and(a, b):
    return a and b

@elementwise('A -> A -> bool')
def logical_or(a, b):
    return a or b

@elementwise('A -> A -> bool')
def logical_xor(a, b):
    return bool(a) ^ bool(b)

@elementwise('A -> bool')
def logical_not(a):
    return not a

#------------------------------------------------------------------------
# Bitwise
#------------------------------------------------------------------------

@elementwise('Array integer N -> Array integer N -> Array integer N')
def bitwise_and(a, b):
    return a & b

@elementwise('Array integer N -> Array integer N -> Array integer N')
def bitwise_or(a, b):
    return a | b

@elementwise('Array integer N -> Array integer N -> Array integer N')
def bitwise_xor(a, b):
    return a ^ b

@elementwise('Array integer N -> Array integer N -> Array integer N')
def left_shift(a, b):
    return a << b

@elementwise('Array integer N -> Array integer N -> Array integer N')
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