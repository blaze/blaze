from ndtable.engine.pipeline import Pipeline
from ndtable.expr.graph import IntNode, FloatNode
from ndtable.expr.visitor import MroTransformer
from ndtable.expr.nodes import flat
from ndtable.table import NDTable, NDArray

from unittest2 import skip
from ndtable.expr.viz import dump

#------------------------------------------------------------------------
# Minivect
#------------------------------------------------------------------------

from ndtable.engine.mv import LazyLLVMContext, miniast, \
   minitypes, specializers, Minivect, get_blaze_pointer

context = LazyLLVMContext()
builder = context.astbuilder

#------------------------------------------------------------------------
# Sample Graph
#------------------------------------------------------------------------

a = NDArray([1])
b = NDArray([1])
x = a+b

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

def test_simple_expr():
    walk = Minivect(context)

    body = walk.visit(x)
    variables = walk.variables

    temp = builder.variable(x.cod.to_minitype(), 'temp')
    prog = builder.assign(temp, body)

    func = builder.build_function(variables, prog, 'func')

    # First arg is the outer shape
    #args = [fist_array.ctypes.shape]
    args = [0]

    for variable in variables:
        if variable.type.is_array:
            numpy_array = variable.value
            data_pointer = get_blaze_pointer(numpy_array, variable.type)
            args.append(data_pointer)
            # Assume contigious so we don't need to specify strides

    specializer = specializers.StridedCInnerContigSpecializer
    specialized_func, (llvm_func, ctypes_func) = specialize(specializer, func)
    ctypes_func(args)
