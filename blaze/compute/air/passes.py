# -*- coding: utf-8 -*-

"""
Passes that massage expression graphs into execution kernels.
"""


from __future__ import absolute_import, division, print_function
from functools import partial

from .prettyprint import verbose
from .frontend import (translate, partitioning, coercions, jit, ckernel_impls,
                       ckernel_lift, allocation, assemblage, ckernel_prepare,
                       ckernel_rewrite)

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
    ckernel_prepare.prepare_local_execution,
    ckernel_impls,
    allocation,
    ckernel_lift,
    ckernel_rewrite,
]

debug_passes = [partial(verbose, p) for p in passes]
