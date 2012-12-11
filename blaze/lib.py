import numpy as np
from blaze.rts.ffi import PythonF, install
from blaze.engine import executors
from numexpr import evaluate

# evaluating this function over a term has a cost, in the future this
# might be things like calculations for FLOPs associated with the
# operation. Could be a function of the shape of the term Most of the
# BLAS functions can calculated a priori

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

install(
    'Add(Array(),Array())',
    PythonF(np.add.types, lambda: None, False),
    zerocost
)

install(
    'Add(<term>,<term>)',
    PythonF(np.add.types, np.add, False),
    zerocost
)


install(
    'Mul(<term>,<term>)',
    PythonF(np.multiply.types, np.multiply, False),
    zerocost
)

install(
    'Pow(<term>,<term>)',
    PythonF(np.power.types, np.power, False),
    zerocost
)

install(
    'Abs(<term>)',
    PythonF(np.abs.types, np.abs, False),
    zerocost
)

# These needn't neccessarily be NumPy functions!

numexpr_add = lambda a,b: evaluate('a+b')

# install(
#     'Add(a,b);*',
#     PythonF(np.add.types, numexpr_add, False),
#     zerocost
# )
