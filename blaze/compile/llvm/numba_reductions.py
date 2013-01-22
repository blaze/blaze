"""
Kernels used for reduction operations, autojitted by numba.
"""

def sum(a, b):
    return a + b

def subtract(a, b):
    return a - b

def prod(a, b):
    return a * b

#def floordiv(a, b):
#    return a // b

def divide(a, b):
    return a / b
