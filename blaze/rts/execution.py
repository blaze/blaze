# a + b * c

# ATerm Graph
# ===========
#
#   Arithmetic(
#     Add
#   , Array(){dshape("3, int64"), 45340864}
#   , Arithmetic(
#         Mul
#       , Array(){dshape("3, int64"), 45340792}
#       , Array(){dshape("3, int64"), 45341584}
#     ){dshape("3, int64"), 45264528}
#   ){dshape("3, int64"), 45264432}

# Execution Plan
# ==============

# vars %a %b %c
# %0 := ElemwiseNumpy{np.mul,nogil}(%b, %c)
# %0 := ElemwiseNumpy{np.add,nogil,inplace}(%0, %a)

# Responsibilities
# - allocate memory blocks on Blaze heap for LHS
# - determine whether to do operation inplace or to store the
#   output in a temporary
#
# - Later: handle kernel fusion
# - Much Later: handle GPU access & thread control

from blaze.rts.heap import Heap

# =================================
# The main Blaze RTS execution loop
# =================================

# Invokes Executor functions and handles memory management from external
# sources to allocate on, IOPro allocators, SQL Queries, ZeroMQ...

# TOOD: write in Cython
def execplan(context, plan, symbols):
    """ Takes a list of of instructions from the Pipeline and
    then allocates the necessary memory needed for the
    intermediates are temporaries """

    h = Heap()
    ret = None

    for instruction in plan:
        ops = [symbols[sym] for sym in symbols]
        dds = [op.asbuflist() for op in ops]
        dss = [op.datashape() for op in ops]

        if instruction.lhs:
            h.allocate(instruction.lhs.size())
            ret = instruction(dds, dss)
        else:
            instruction(dds, dss)

    h.finalize()
    return ret
