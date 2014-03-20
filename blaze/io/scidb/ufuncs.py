"""SciDB implementations element-wise ufuncs."""

from __future__ import absolute_import, division, print_function

from blaze.compute.ops import ufuncs
from blaze.compute.ops.ufuncs import not_equal, less, logical_not
from .kernel import SCIDB

from .query import apply, iff, qformat


def overload_unop_ufunc(signature, name, op):
    """Add a unary sql overload to a blaze ufunc"""
    def unop(x):
        return apply_expr(x, qformat('{op} {x}.f0', op=op, x=x))
    unop.__name__ = name
    bf = getattr(ufuncs, name)
    bf.add_plugin_overload(signature, unop, SCIDB)


def overload_binop_ufunc(signature, name, op):
    """Add a binary sql overload to a blaze ufunc"""
    def binop(a, b):
        arr = qformat("join({a}, {b})", a=a, b=b)
        expr = qformat("{a}.f0 {op} {b}.f0", a=a, op=op, b=b)
        return apply_expr(arr, expr)
    binop.__name__ = name
    bf = getattr(ufuncs, name)
    bf.add_plugin_overload(signature, binop, SCIDB)


#------------------------------------------------------------------------
# Arithmetic
#------------------------------------------------------------------------
overload_binop_ufunc("(T, T) -> T", "add", "+")
overload_binop_ufunc("(T, T) -> T", "multiply", "*")
#overload_binop_ufunc("(A : real, A) -> A", "subtract", "-")
overload_binop_ufunc("(T, T) -> T", "subtract", "-")
#overload_binop_ufunc("(A : real, A) -> A", "divide", "/")
overload_binop_ufunc("(T, T) -> T", "divide", "/")
#overload_binop_ufunc("(A : real, A) -> A", "mod", "%")
overload_binop_ufunc("(T, T) -> T", "mod", "%")

# overload_binop_ufunc("(A : real, A) -> A", "floordiv", "//")
# overload_binop_ufunc("(A : real, A) -> A", "truediv", "/")

overload_unop_ufunc("(T) -> T", "negative", "-")

#------------------------------------------------------------------------
# Compare
#------------------------------------------------------------------------
overload_binop_ufunc("(T, T) -> bool", "equal", "==")
overload_binop_ufunc("(T, T) -> bool", "not_equal", "!=")
overload_binop_ufunc("(T, T) -> bool", "less", "<")
overload_binop_ufunc("(T, T) -> bool", "greater", ">")
overload_binop_ufunc("(T, T) -> bool", "greater_equal", ">=")

#------------------------------------------------------------------------
# Logical
#------------------------------------------------------------------------
# TODO: We have to implement all combinations of types here for 'and' etc,
#       given the set {numeric, bool} for both arguments. Overloading at the
#       kernel level would reduce this. Can we decide between "kernels" and
#       "functions" depending on the inference process.

# TODO: numeric/bool and bool/numeric combinations


def logical_and(a, b):
    return iff(ibool(a). ibool(b), false)
ufuncs.logical_and.add_plugin_overload("(T, T) -> bool",
                                       logical_and, SCIDB)


def logical_and(a, b):
    return iff(a, b, false)
ufuncs.logical_and.add_plugin_overload("(bool, bool) -> bool",
                                       logical_and, SCIDB)


def logical_or(a, b):
    return iff(ibool(a), true, ibool(b))
ufuncs.logical_or.add_plugin_overload("(T, T) -> bool",
                                      logical_or, SCIDB)

def logical_or(a, b):
    return iff(a, true, b)
ufuncs.logical_or.add_plugin_overload("(bool, bool) -> bool",
                                      logical_or, SCIDB)


# Fixme: repeat of subexpression leads to exponential code generation !


def logical_xor(a, b):
    return iff(ibool(a), logical_not(ibool(b)), ibool(b))
ufuncs.logical_xor.add_plugin_overload("(T, T) -> bool",
                                       logical_xor, SCIDB)


def logical_xor(a, b):
    return iff(a, logical_not(b), b)
ufuncs.logical_xor.add_plugin_overload("(bool, bool) -> bool",
                                       logical_xor, SCIDB)


def logical_not(a):
    return apply("not", a)
ufuncs.logical_not.add_plugin_overload("(T) -> bool",
                                       logical_not, SCIDB)


#------------------------------------------------------------------------
# Math
#------------------------------------------------------------------------


def abs(x):
    # Fixme: again exponential codegen
    return iff(less(x, 0), ufuncs.negative(x), x)
ufuncs.abs.add_plugin_overload("(T) -> bool",
                                       abs, SCIDB)


#------------------------------------------------------------------------
# Helper functions
#------------------------------------------------------------------------
def ibool(x):
    return not_equal(x, "0")


def apply_expr(arr, expr):
    colname = '__blaze_col'
    query = qformat('apply({arr}, {colname}, {expr})',
                    arr=arr, colname=colname, expr=expr)
    return project(query, colname)


def project(arr, colname):
    return qformat('project({arr}, {colname})', arr=arr, colname=colname)

#------------------------------------------------------------------------
# Data types
#------------------------------------------------------------------------

true = "true"
false = "false"
