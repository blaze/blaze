import numpy as np
from math import sqrt

from blaze.datashape import dshape
from blaze.rts.ffi import PythonFn, install, lift
from blaze.engine import executors
from numexpr import evaluate
from blaze.ts.ucr_dtw import ucr

# evaluating this function over a term has a cost, in the future this
# might be things like calculations for FLOPs associated with the
# operation. Could be a function of the shape of the term Most of the
# costs of BLAS functions can calculated a priori

zerocost = lambda term: 0

#------------------------------------------------------------------------
# Preinstalled Functions
#------------------------------------------------------------------------

# These are side-effectful functions which install the core
# functions into the RTS dispatcher.

# The signature for PythonF

#   :signature: ATerm pattern matching signature

#   :fn: Python callable instance

#   :mayblock: Whether calling this function may block a thread.
#              ( i.e. it waits on a disk or socket )


# TODO: right now these consume everything but later we'll add functions
# which specialize on metadata for contigious, chunked, streams,
# etc...

@lift('Sqrt(<int>)', 'a -> float32')
def pyadd(a):
    return sqrt(a)

@lift('dtw(Array(), Array(), <term>, <term>)', '(a,a,float,int) -> b')
def dtw(d1, d2, s, n):
    return ucr.dtw(d1, d2, s, n)


install(
    'Add(<term>,<term>)',
    PythonFn(np.add.types, np.add, False),
    zerocost
)

install(
    'Mul(<term>,<term>)',
    PythonFn(np.multiply.types, np.multiply, False),
    zerocost
)

install(
    'Pow(<term>,<term>)',
    PythonFn(np.power.types, np.power, False),
    zerocost
)

install(
    'Abs(<term>)',
    PythonFn(np.abs.types, np.abs, False),
    zerocost
)

# ==============
# --- Future ---
# ==============

# Specialized just for arrays
# ---------------------------
#
# install(
#     'Add(Array(),Array())',
#     PythonF(np.add.types, np.add, False),
#     zerocost
# )

# Specialized just for contigious arrays
# --------------------------------------
#
# install(
#     'Add(Array(){contigious},Array(){contigious})',
#     PythonF(np.add.types, np.add, False),
#     zerocost
# )

# These also needn't neccessarily be NumPy functions!

numexpr_add = lambda a,b: evaluate('a+b')

# install(
#     'Add(a,b);*',
#     PythonF(np.add.types, numexpr_add, False),
#     zerocost
# )
