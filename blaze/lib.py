import numpy as np
from blaze.rts.ffi import PythonF, install

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

install(
    'Add(a,b) ; contig',
    PythonF(np.add.types, np.add, False),
    zerocost
)

install(
    'Mul(a,b) ; contig',
    PythonF(np.multiply.types, np.multiply, False),
    zerocost
)

install(
    'Abs(a) ; contig',
    PythonF(np.abs.types, np.abs, False),
    zerocost
)
