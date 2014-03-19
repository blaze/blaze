"""IR definitions."""

from __future__ import print_function, division, absolute_import

import math
import operator

import numpy as np

from . import ops
from .utils import invert, mergedicts

#===------------------------------------------------------------------===
# Python Version Compatibility
#===------------------------------------------------------------------===

def divide(a, b):
    """
    `a / b` with python 2 semantics:

        - floordiv() integer division
        - truediv() float division
    """
    if isinstance(a, (int, long)) and isinstance(b, (int, long)):
        return operator.floordiv(a, b)
    else:
        return operator.truediv(a, b)

def erfc(x):
    # Python 2.6
    # libm = ctypes.util.find_library("m")
    # return libm.erfc(x)

    return math.erfc(x)

#===------------------------------------------------------------------===
# Definitions -> Evaluation function
#===------------------------------------------------------------------===

unary = {
    ops.invert        : operator.inv,
    ops.not_          : operator.not_,
    ops.uadd          : operator.pos,
    ops.usub          : operator.neg,
}

binary = {
    ops.add           : operator.add,
    ops.sub           : operator.sub,
    ops.mul           : operator.mul,
    ops.div           : divide,
    ops.mod           : operator.mod,
    ops.lshift        : operator.lshift,
    ops.rshift        : operator.rshift,
    ops.bitor         : operator.or_,
    ops.bitand        : operator.and_,
    ops.bitxor        : operator.xor,
}

compare = {
    ops.lt            : operator.lt,
    ops.le            : operator.le,
    ops.gt            : operator.gt,
    ops.ge            : operator.ge,
    ops.eq            : operator.eq,
    ops.ne            : operator.ne,
    ops.is_           : operator.is_,
    #ops.contains      : operator.contains,
}

math_funcs = {
    ops.Sin         : np.sin,
    ops.Asin        : np.arcsin,
    ops.Sinh        : np.sinh,
    ops.Asinh       : np.arcsinh,
    ops.Cos         : np.cos,
    ops.Acos        : np.arccos,
    ops.Cosh        : np.cosh,
    ops.Acosh       : np.arccosh,
    ops.Tan         : np.tan,
    ops.Atan        : np.arctan,
    ops.Atan2       : np.arctan2,
    ops.Tanh        : np.tanh,
    ops.Atanh       : np.arctanh,
    ops.Log         : np.log,
    ops.Log2        : np.log2,
    ops.Log10       : np.log10,
    ops.Log1p       : np.log1p,
    ops.Exp         : np.exp,
    ops.Exp2        : np.exp2,
    ops.Expm1       : np.expm1,
    ops.Floor       : np.floor,
    ops.Ceil        : np.ceil,
    ops.Abs         : np.abs,
    ops.Erfc        : erfc,
    ops.Rint        : np.rint,
    ops.Pow         : np.power,
    ops.Round       : np.round,
}

#===------------------------------------------------------------------===
# Definitions
#===------------------------------------------------------------------===

unary_defs = {
    "~": ops.invert,
    "!": ops.not_,
    "+": ops.uadd,
    "-": ops.usub,
}

binary_defs = {
    "+":  ops.add,
    "-":  ops.sub,
    "*":  ops.mul,
    "/":  ops.div,
    "%":  ops.mod,
    "<<": ops.lshift,
    ">>": ops.rshift,
    "|":  ops.bitor,
    "&":  ops.bitand,
    "^":  ops.bitxor,
}

compare_defs = {
    "<":  ops.lt,
    "<=": ops.le,
    ">":  ops.gt,
    ">=": ops.ge,
    "==": ops.eq,
    "!=": ops.ne,
}

unary_opcodes = invert(unary_defs)
binary_opcodes = invert(binary_defs)
compare_opcodes = invert(compare_defs)

opcode2operator = mergedicts(unary, binary, compare)
operator2opcode = mergedicts(invert(unary), invert(binary), invert(compare))
bitwise = set(["<<", ">>", "|", "&", "^", "~"])
