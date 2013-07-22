from __future__ import absolute_import

"""
The bkernel submodule of Blaze implements kernels for JIT
compilation with LLVM.
"""

from .blaze_kernels import BlazeElementKernel
from .blfuncs import KernelTree, BlazeFunc
