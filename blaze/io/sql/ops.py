"""SQL implementations of element-wise ufuncs."""

from __future__ import absolute_import, division, print_function

from ...compute.function import BlazeFunc
from ...compute.ops import ufuncs
from .kernel import SQL
from .syntax import Call, Expr, QOrderBy, QWhere, And, Or, Not


def sqlfunction(signature):
    def decorator(f):
        bf = BlazeFunc('blaze', f.__name__)
        # FIXME: Adding a dummy CKERNEL overload to make things work for now
        bf.add_overload(signature, None)
        bf.add_plugin_overload(signature, f, SQL)
        return bf
    return decorator


def overload_unop_ufunc(signature, name, op):
    """Add a unary sql overload to a blaze ufunc"""
    def unop(x):
        return Expr([op, x])
    unop.__name__ = name
    bf = getattr(ufuncs, name)
    bf.add_plugin_overload(signature, unop, SQL)


def overload_binop_ufunc(signature, name, op):
    """Add a binary sql overload to a blaze ufunc"""
    def binop(a, b):
        return Expr([a, op, b])
    binop.__name__ = name
    bf = getattr(ufuncs, name)
    bf.add_plugin_overload(signature, binop, SQL)


# Arithmetic

overload_binop_ufunc("(T, T) -> T", "add", "+")
overload_binop_ufunc("(T, T) -> T", "multiply", "*")
overload_binop_ufunc("(T, T) -> T", "subtract", "-")
overload_binop_ufunc("(T, T) -> T", "floor_divide", "/")
overload_binop_ufunc("(T, T) -> T", "divide", "/")
overload_binop_ufunc("(T, T) -> T", "true_divide", "/")
overload_binop_ufunc("(T, T) -> T", "mod", "%")

overload_unop_ufunc("(T) -> T", "negative", "-")

# Compare

overload_binop_ufunc("(T, T) -> bool", "equal", "==")
overload_binop_ufunc("(T, T) -> bool", "not_equal", "!=")
overload_binop_ufunc("(T, T) -> bool", "less", "<")
overload_binop_ufunc("(T, T) -> bool", "less_equal", "<=")
overload_binop_ufunc("(T, T) -> bool", "greater", ">")
overload_binop_ufunc("(T, T) -> bool", "greater_equal", ">=")

# Logical

overload_binop_ufunc("(bool, bool) -> bool",
                     "logical_and", "AND")
overload_binop_ufunc("(bool, bool) -> bool",
                     "logical_or", "OR")
overload_unop_ufunc("(bool) -> bool", "logical_not", "NOT")


def logical_xor(a, b):
    # Potential exponential code generation...
    return And(Or(a, b), Not(And(a, b)))

ufuncs.logical_xor.add_plugin_overload("(bool, bool) -> bool",
                                       logical_xor, SQL)

# SQL Functions

@sqlfunction('(A * DType) -> DType')
def sum(col):
    return Call('SUM', [col])

@sqlfunction('(A * DType) -> DType')
def avg(col):
    return Call('AVG', [col])

@sqlfunction('(A * DType) -> DType')
def min(col):
    return Call('MIN', [col])

@sqlfunction('(A * DType) -> DType')
def max(col):
    return Call('MAX', [col])

# SQL Join, Where, Group by, Order by

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


@sqlfunction('(A * S, A * B) -> var * S')
def sqlindex(col, where):
    return QWhere(col, where)

@sqlfunction('(A * S, A * B) -> A * S')
def sqlorder(col, by):
    if not isinstance(by, (tuple, list)):
        by = [by]
    return QOrderBy(col, by)
