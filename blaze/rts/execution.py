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

def execplan(context, plan):
    """ Take an execution plan and execute it """
    pass
