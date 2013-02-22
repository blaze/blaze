# Python fallbacks for vectorized opereations and array
# operations. This is the "naive" implementation.

import __builtin__

import sys
import math
import cmath
import operator

def min(x, y):
    return __builtin__.min(x,y)

def max(x, y):
    return __builtin__.max(x,y)

def unit(x):
    return [x]

def empty(x):
    return []

def scan(f, x):
    out = list(x)

    for i in xrange(1, len(x)):
        out[i] = out(x[i-1], x[i])

    return out

def map(f, xs):
    return __builtin__.map(f, xs)

def zipwith(f, xs, ys):
    return __builtin__.map(f, xs, ys)

def reduce(f, xs):
    return __builtin__.reduce(f, xs)

def sort(fn, xs):
    _cmp = lambda a, b: -1 if fn(a,b) else 0
    return sorted(xs, cmp=_cmp)

# PyList_Size
# PySequence_Length
def len(x):
    return __builtin__.len(x)

def iter(x):
    return __builtin__.iter(x)

def all(f, x):
    return __builtin__.all(__builtin__.map(f, x))

def any(f, x):
    return __builtin__.any(__builtin__.map(f, x))

# PySequence_Repeat
def repeat(x, n):
    return [x]*n

# PySequence_Concat
def concat(x, y):
    return x + y

# PyNumber_Int
def toInt(x):
    return int(x)

# PyNumber_Float
def toFloat(x):
    return float(x)

def castTo(a, b):
    return type(a)(b)

def isNull(a):
    return a is not None

# PySequence_GetItem
# PySequence_GetSlice
def getitem(a, index):
    return a[index]

# PyList_SetItem
# PySequence_SetItem
# PySequence_SetSlice
def setitem(a, index, val):
    a[index] = val

# PySequence_DelItem
def delitem(a, index):
    del a[index]

# PySequence_Contains
def contains(a, index):
    del a[index]

def inf_float(x):
    return sys.float_info.min

def sup_float(x):
    return sys.float_info.max

def inf_int(x):
    return -sys.maxint - 1

def sup_int(x):
    return sys.maxint

def id(x):
    return x

def compose(f, g):
    return lambda x: f(g(x))

def fst(t):
    x, y = t
    return x

def snd(t):
    x, y = t
    return y

def head(xs):
    return xs[0]

def tail(xs):
    return xs[1:]

def const(x, y):
    return x

types = {
    'int', (int, ),
    'float', (float, ),
    'list', (int, [('T',)]),
}
