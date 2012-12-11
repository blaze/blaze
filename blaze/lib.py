import numpy as np
from blaze.rts.ffi import PythonF, install
from blaze.engine import executors

zerocost = lambda x: 0

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
    'Add(a,b);*',
    PythonF(np.add.types, np.add, False),
    zerocost
)

install(
    'Mul(a,b);*',
    PythonF(np.multiply.types, np.multiply, False),
    zerocost
)

install(
    'Pow(a,b);*',
    PythonF(np.power.types, np.power, False),
    zerocost
)

install(
    'Abs(a);*',
    PythonF(np.abs.types, np.abs, False),
    zerocost
)
