# a + b * c

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

# Invokes Executor functions and handles memory management from external
# sources to allocate on, IOPro allocators, SQL Queries, ZeroMQ...

from blaze.rts.heap import Heap

# Write now we're trying to emphasize the protocol semantics,
# not performance!

# TODO: Write in C or Cython.

# Blaze X -> X
def execute(context, vartable, instructions):
    """ Takes a list of of instructions from the Pipeline and
    then allocates the necessary memory needed for the
    intermediates are temporaries """

    h = Heap()
    ret = None

    for instruction in instruction:
        ops = [vartable[uri] for uri in vartable]
        dds = [op.asbuflist() for op in ops]
        dss = [op.datashape() for op in ops]

        if instruction.lhs:
            h.allocate(instruction.lhs.size())
            ret = instruction(dds, dss)
        else:
            instruction(dds, dss)

    h.finalize()
    return ret
