"""SciDB implementations element-wise ufuncs."""

from __future__ import absolute_import, division, print_function

from blaze.compute.ops import ufuncs
from blaze.compute.ops.ufuncs import not_equal, less, logical_not

from .kernel import scidb_elementwise, scidb_kernel
from .query import apply, iff, qformat


#------------------------------------------------------------------------
# Implement functions
#------------------------------------------------------------------------
def define_unop(signature, name, op):
    """Define a unary scidb operator"""
    def unop(x):
        return apply_expr(x, qformat('{op} {x}.f0', op=op, x=x))

    unop.__name__ = name
    _implement(unop, signature)
    return unop


def define_binop(signature, name, op):
    """Define a binary scidb operator"""
    def binop(a, b):
        arr = qformat("join({a}, {b})", a=a, b=b)
        expr = qformat("{a}.f0 {op} {b}.f0", a=a, op=op, b=b)
        return apply_expr(arr, expr)

    binop.__name__ = name
    _implement(binop, signature)
    return binop


def _implement(f, signature):
    name = f.__name__
    blaze_func = getattr(ufuncs, name)
    #print("implement", f, signature, blaze_func)
    scidb_kernel(blaze_func, f, signature)


#------------------------------------------------------------------------
# Arithmetic
#------------------------------------------------------------------------
add = define_binop("(A... * T, A... * T) -> A... * T", "add", "+")
multiply = define_binop("(A... * T, A... * T) -> A... * T", "multiply", "*")
#subtract = define_binop("(A : real, A) -> A", "subtract", "-")
subtract = define_binop("(A... * T, A... * T) -> A... * T", "subtract", "-")
#divide = define_binop("(A : real, A) -> A", "divide", "/")
divide = define_binop("(A... * T, A... * T) -> A... * T", "divide", "/")
#mod = define_binop("(A : real, A) -> A", "mod", "%")
mod = define_binop("(A... * T, A... * T) -> A... * T", "mod", "%")

# floordiv = define_binop("(A : real, A) -> A", "floordiv", "//")
# truediv = define_binop("(A : real, A) -> A", "truediv", "/")

negative = define_unop("(A... * T) -> A... * T", "negative", "-")

#------------------------------------------------------------------------
# Compare
#------------------------------------------------------------------------
equal = define_binop("(A... * T, A... * T) -> A... * bool", "add", "==")
not_equal = define_binop("(A... * T, A... * T) -> A... * bool", "add", "!=")
less = define_binop("(A... * T, A... * T) -> A... * bool", "add", "<")
less_equal = define_binop("(A... * T, A... * T) -> A... * bool", "add", "<=")
greater = define_binop("(A... * T, A... * T) -> A... * bool", "add", ">")
greater_equal = define_binop("(A... * T, A... * T) -> A... * bool", "add", ">=")

#------------------------------------------------------------------------
# Logical
#------------------------------------------------------------------------
# TODO: We have to implement all combinations of types here for 'and' etc,
#       given the set {numeric, bool} for both arguments. Overloading at the
#       kernel level would reduce this. Can we decide between "kernels" and
#       "functions" depending on the inference process.

# TODO: numeric/bool and bool/numeric combinations

# --- and ---

#@scidb_elementwise('(A... * T : numeric, A... * T) -> A... * bool')
@scidb_elementwise('(A... * T, A... * T) -> A... * bool')
def logical_and(a, b):
    return iff(ibool(a). ibool(b), false)

@scidb_elementwise('(A... * bool, A... * bool) -> A... * bool')
def logical_and(a, b):
    return iff(a, b, false)

# --- or ---

#@scidb_elementwise('(A... * T : numeric, A... * T) -> A... * bool')
@scidb_elementwise('(A... * T, A... * T) -> A... * bool')
def logical_or(a, b):
    return iff(ibool(a), true, ibool(b))

@scidb_elementwise('(A... * bool, A... * bool) -> A... * bool')
def logical_or(a, b):
    return iff(a, true, b)

# --- xor ---

# Fixme: repeat of subexpression leads to exponential code generation !

#@scidb_elementwise('(A... * T : numeric, A... * T) -> A... * bool')
@scidb_elementwise('(A... * T, A... * T) -> A... * bool')
def logical_xor(a, b):
    return iff(ibool(a), logical_not(ibool(b)), ibool(b))

@scidb_elementwise('(A... * bool, A... * bool) -> A... * bool')
def logical_xor(a, b):
    return iff(a, logical_not(b), b)

# --- not ---

@scidb_elementwise('(A... * T) -> A... * bool')
def logical_not(a):
    return apply("not", a)


#------------------------------------------------------------------------
# Math
#------------------------------------------------------------------------
#@scidb_elementwise('(A : numeric) -> A')
@scidb_elementwise('(A... * T) -> A... * T')
def abs(x):
    # Fixme: again exponential codegen
    return iff(less(x, 0), negative(x), x)


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
