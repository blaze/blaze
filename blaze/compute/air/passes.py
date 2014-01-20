# -*- coding: utf-8 -*-

"""
Passes that massage expression graphs into execution kernels.
"""

from __future__ import absolute_import, division, print_function
from .frontend import (translate, coercions, jit, ckernel_impls,
                       ckernel_lift, allocation)

#------------------------------------------------------------------------
# Passes
#------------------------------------------------------------------------

passes = [
    translate,
    # cache, # TODO:
    coercions,
    jit,
    # TODO: Make the below compile-time passes !
    #ckernel_impls,
    #allocation,
    #ckernel_lift,

    # assemble_kernels, # TODO:
]