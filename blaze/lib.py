"""

Anatomy of a Blaze Function Def
-------------------------------

::

       ATerm Pattern Matcher
                |                      +-- Type Signature
                |                      |
                v                      v
     @lift('Mul(<term>,<term>)', '(a,a) -> a', {
         'types'   : {'a': array_like},     <- Type Constraints
         'metadata': {'a': aligned, local}, <- Metadata Constraint
     }, costfn)
     def multiply(a, b):
         return np.multipy(a, b)
        |                     |
        +---------------------+
        Function Implementation


If we were to "read" this definition in English. It would read:

> Here is a function called Mul, it matches any expression of two
> graph nodes. In addition the two graph nodes must be of the same
> datashapes (a) or unify/broadcast to the same type (a,a). It
> returns a result of the same type of the operands. In addition the
> two arguments must have metadata annotating them both as aligned
> memory and in system local memory. If the expression in question
> matches these criterion than I can tell you that you can perfom
> this operation in (n*m) FLOPS where n and m are the size of your
> input array, if you can't find a better implementaion than that use
> this function implementation!

"""

import numpy as np

from blaze.funcs import lift
from blaze import metadata as md

from blaze.metadata import aligned
from blaze.expr.ops import array_like, table_like

from blaze.ts.ucr_dtw import ucr
#from blaze.ooc import linalg
from blaze.algo import stats
from blaze.algo import select as _select

zerocost = lambda term: 0

#------------------------------------------------------------------------
# Statistics
#------------------------------------------------------------------------

@lift('mean(<term>, <str>)', '(a,_) -> a', dict(
    types =  {'a': table_like},
    metadata = {},
    passthrough = True
))
def mean(table, label):
    return stats.mean(table, label)

@lift('std(<term>, <str>)', '(a,_) -> a', dict(
    types =  {'a': table_like},
    metadata = {},
    passthrough = True
))
def std(table, label):
    return stats.std(table, label)

@lift('select(<term>, <str>)', 'a -> a', dict(
    passthrough = True
))
def select(table, predicate, label):
    return _select(table, predicate, label)

#------------------------------------------------------------------------
# Time Series
#------------------------------------------------------------------------

@lift('dtw(<term>, <term>, <term>, <term>)', '(a,a,float,int) -> b', {
    'types' : {'a': array_like},
    'metadata': {},
})
def dtw(d1, d2, s, n):
    return ucr.dtw(d1, d2, s, n)

#------------------------------------------------------------------------
# Linear Algebra
#------------------------------------------------------------------------

@lift('dot(<term>, <term>)', '(a,a) -> a', {
    'types'    : {'a' : array_like},
    'metadata' : {'a' : md.c_contigious},
})
def dot(a1, a2):
    return linalg.dot(a1, a2)

#------------------------------------------------------------------------
# Arithmetic
#------------------------------------------------------------------------

@lift('Add(<term>,<term>)', '(a,a) -> a', {})
def add(a, b):
    from blaze import zeros

    ds_a, dd_a = a
    ds_b, dd_b = b
    out = zeros(ds_a)

    iters = [desc.as_chunked_iterator() for desc in [dd_a, dd_b]]
    for a,b in zip(*iters):
        pass
    return out


@lift('Mul(<term>,<term>)', '(a,a)-> a', {
    'types'   : {'a': array_like},
    'metadata': {},
})
def multiply(a, b):
    return np.multipy(a, b)


@lift('Pow(<term>,<term>)', '(a,a) -> a', {
    'types'   : {'a': array_like},
    'metadata': {},
})
def power(a, b):
    return np.power(a, b)


@lift('Abs(<term>)', 'a -> a', {
    'types'   : {'a': array_like},
    'metadata': {},
})
def abs(a, b):
    return np.abs(a, b)


__all__ = [
    'lift',
    'mean',
    'std',
    'select',
    'abs',
    'dot',
    'dtw',
    'add',
    'multiply',
    'power'
]
