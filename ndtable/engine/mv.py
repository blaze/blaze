import sys
import time

import numpy as np

from minivect import miniast
from minivect import specializers
from minivect import minitypes
from minivect.ctypes_conversion import get_data_pointer, \
    get_pointer, convert_to_ctypes
from ndtable.datashape.coretypes import var_generator

from ndtable.expr.visitor import MroTransformer

context_debug = 0

class LazyLLVMContext(miniast.LLVMContext):
    debug = context_debug
    def stridesvar(self, variable):
        return miniast.StridePointer(self.pos, minitypes.NPyIntp.pointer(),
                                     variable)

#------------------------------------------------------------------------
# Blaze Pipeline
#------------------------------------------------------------------------

def setup(debug=0):
    context = LazyLLVMContext()
    builder = context.astbuilder

    ccontext = miniast.CContext()
    ccontext.debug = debug

    return context, ccontext, builder

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def get_array_pointer(numpy_array, array_type):
    dtype_pointer = array_type.dtype.pointer()
    return numpy_array.ctypes.data_as(convert_to_ctypes(dtype_pointer))

def specialize(specializer_cls, ast, context):
    specializers = [specializer_cls]
    result = iter(context.run(ast, specializers)).next()
    _, specialized_ast, _, code_output = result
    return specialized_ast, code_output

#------------------------------------------------------------------------
# Mapper
#------------------------------------------------------------------------

class Minivect(MroTransformer):
    """
    Map between the Blaze graph objects into the Minivect AST.
    Accumulating the visited variables statefully.
    """

    def __init__(self, context):
        self.builder = context.astbuilder
        self.variables = []

    def ArrayNode(self, node):
        return node

    def App(self, node):
        lhs, rhs = self.visit(node.children)[0]
        op = node.operator.op
        return self.builder.binop(lhs.type, op, lhs, rhs)

    def BinaryOp(self, node):
        lhs, rhs = map(self.visit, node.children)

        if isinstance(lhs, list):
            lhs = self.visit(lhs)

        if isinstance(rhs, list):
            rhs = self.visit(rhs)

        return lhs, rhs

    def Literal(self, node):
        minidtype = node.datashape.to_minitype()
        variable = self.builder.variable(minidtype, str(id(node)))
        variable.value = node.val
        self.variables.append(variable)
        return variable
