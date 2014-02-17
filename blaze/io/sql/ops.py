"""SQL implementations of element-wise ufuncs."""

from __future__ import absolute_import, division, print_function

from ...compute.function import function, kernel
from ...compute.ops import ufuncs
from .kernel import sql_kernel, SQL
from .syntax import Call, Expr, QOrderBy, QGroupBy, QWhere, And, Or, Not

def sqlfunction(signature):
    def decorator(f):
        blaze_func = function(signature)(f)
        kernel(blaze_func, SQL, f, signature)
        return blaze_func
    return decorator

#------------------------------------------------------------------------
# Implement functions
#------------------------------------------------------------------------

def define_unop(signature, name, op):
    """Define a unary sql operator"""
    def unop(x):
        return Expr([op, x])
    unop.__name__ = name
    _implement(unop, signature)
    return unop


def define_binop(signature, name, op):
    """Define a binary sql operator"""
    def binop(a, b):
        return Expr([a, op, b])
    binop.__name__ = name
    _implement(binop, signature)
    return binop


def _implement(f, signature):
    name = f.__name__
    blaze_func = getattr(ufuncs, name)
    #print("implement", f, signature, blaze_func)
    sql_kernel(blaze_func, f, signature)

#------------------------------------------------------------------------
# Arithmetic
#------------------------------------------------------------------------

add = define_binop("a -> a -> a", "add", "+")
multiply = define_binop("a -> a -> a", "multiply", "*")
subtract = define_binop("a -> a -> a", "subtract", "-")
floordiv = define_binop("a -> a -> a", "floor_divide", "/")
divide = define_binop("a -> a -> a", "divide", "/")
truediv = define_binop("a -> a -> a", "true_divide", "/")
mod = define_binop("a -> a -> a", "mod", "%")

negative = define_unop("a -> a", "negative", "-")

#------------------------------------------------------------------------
# Compare
#------------------------------------------------------------------------

eq = define_binop("a..., T -> a..., T -> a..., bool", "equal", "==")
ne = define_binop("a..., T -> a..., T -> a..., bool", "not_equal", "!=")
lt = define_binop("a..., T -> a..., T -> a..., bool", "less", "<")
le = define_binop("a..., T -> a..., T -> a..., bool", "less_equal", "<=")
gt = define_binop("a..., T -> a..., T -> a..., bool", "greater", ">")
ge = define_binop("a..., T -> a..., T -> a..., bool", "greater_equal", ">=")

#------------------------------------------------------------------------
# Logical
#------------------------------------------------------------------------

logical_and = define_binop("a..., bool -> a..., bool -> a..., bool",
                           "logical_and", "AND")
logical_or  = define_binop("a..., bool -> a..., bool -> a..., bool",
                           "logical_or", "OR")
logical_not = define_unop("a..., bool -> a..., bool", "logical_not", "NOT")

def logical_xor(a, b):
    # Potential exponential code generation...
    return And(Or(a, b), Not(And(a, b)))

kernel(ufuncs.logical_xor, SQL, logical_xor,
       "a..., bool -> a..., bool -> a..., bool")

#------------------------------------------------------------------------
# SQL Functions
#------------------------------------------------------------------------

# TODO: AVG, MIN, MAX, SUM, ...

def merge(left, right, how='left', on=None, left_on=None, right_on=None,
          left_index=False, right_index=False, sort=True):
    """
    Join two tables.
    """
    raise NotImplementedError


def index(col, index, order=None):
    """
    Index a table or column with a predicate.

        view = merge(table1, table2)
        result = view[table1.id == table2.id]

    or

        avg(table1.age[table1.state == 'TX'])
    """
    result = sqlindex(col, index)
    if order:
        result = sqlorder(result, order)
    return result


@sqlfunction('a -> b -> a')
def sqlindex(col, where):
    return QWhere(col, where)

@sqlfunction('a -> b -> a')
def sqlorder(col, by):
    if not isinstance(by, (tuple, list)):
        by = [by]
    return QOrderBy(col, by)