"""
Blaze element-wise ufuncs.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['add', 'multiply', 'subtract', 'divide', 'true_divide',
           'floor_divide', 'mod', 'negative',
           'equal', 'not_equal', 'less', 'less_equal', 'greater',
           'greater_equal',
           'logical_or', 'logical_and', 'logical_xor', 'logical_not',
           'bitwise_and', 'bitwise_or', 'bitwise_xor',
           'left_shift', 'right_shift',
           'abs']

try:
    import __builtin__ as builtins
except ImportError:
    import builtins

from ..function import jit_elementwise

@jit_elementwise('a -> a -> a')
def add(a, b):
    return a + b

@jit_elementwise('a -> a -> a')
def multiply(a, b):
    return a * b

@jit_elementwise('a -> a -> a')
def subtract(a, b):
    return a - b

@jit_elementwise('a -> a -> a')
def divide(a, b):
    return a / b

@jit_elementwise('a -> a -> a')
def true_divide(a, b):
    return a / b

@jit_elementwise('a -> a -> a')
def floor_divide(a, b):
    return a // b

@jit_elementwise('a -> a -> a')
def mod(a, b):
    return a % b

@jit_elementwise('a -> a')
def negative(a):
    return -a

#------------------------------------------------------------------------
# Compare
#------------------------------------------------------------------------

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def equal(a, b):
    return a == b

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def not_equal(a, b):
    return a != b

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def less(a, b):
    return a < b

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def less_equal(a, b):
    return a <= b

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def greater(a, b):
    return a > b

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def greater_equal(a, b):
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

#------------------------------------------------------------------------
# Math
#------------------------------------------------------------------------

@jit_elementwise('A -> A')
def abs(x):
    return builtins.abs(x)
