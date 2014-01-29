# -*- coding: utf-8 -*-

"""
Passes that massage expression graphs into execution kernels.
"""

from __future__ import absolute_import, division, print_function
from .frontend import (translate, partitioning, coercions, jit, ckernel_impls,
                       ckernel_lift, allocation, assemblage)
from .execution import jit_interp

#------------------------------------------------------------------------
# Passes
#------------------------------------------------------------------------

passes = [
    translate,

    partitioning.annotate_all_kernels,
    partitioning.partition,
    #partitioning.annotate_kernels,
    partitioning.annotate_roots,

    # erasure, # TODO: erase shape from ops
    # cache, # TODO:
    coercions,
    jit,
    assemblage.assemble_py_kernels,
    # TODO: Make the below compile-time passes !
    #ckernel_impls,
    #allocation,
    #ckernel_lift,

    #assemblage.assemble_py_kernels,
]