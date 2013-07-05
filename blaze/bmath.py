#bmath.py
from blaze.blfuncs import BlazeFunc

def make_blazefunc(f):
    return BlazeFunc(f.__name__, template=f)

@make_blazefunc
def add(a, b):
    return a + b

@make_blazefunc
def mul(a, b):
    return a * b

@make_blazefunc
def sub(a, b):
    return a - b

@make_blazefunc
def div(a, b):
    return a / b

@make_blazefunc
def truediv(a, b):
    return a / b

@make_blazefunc
def floordiv(a, b):
    return a // b

@make_blazefunc
def mod(a, b):
    return a % b

@make_blazefunc
def eq(a, b):
    return a == b

@make_blazefunc
def ne(a, b):
    return a != b

@make_blazefunc
def lt(a, b):
    return a < b

@make_blazefunc
def le(a, b):
    return a <= b

@make_blazefunc
def gt(a, b):
    return a > b

@make_blazefunc
def ge(a, b):
    return a >= b

