"""
SciDB query generation and execution from Blaze AIR.
"""

from __future__ import absolute_import, division, print_function

from pykit.ir import interp

import blaze
from blaze.io.sql import SQL

from .error import sqlerror
from .query import execute, dynd_chunk_iterator
from .datadescriptor import SQLDataDescriptor

#------------------------------------------------------------------------
# Interpreter
#------------------------------------------------------------------------

# TODO: Make partitioning generic for any backend

def partition_sql(func, env):
    """
    Determine all operations that are supposed to be executed in SQL land.
    """
    md = env['inputs.metadata']
    sql_ops = set()

    for arg in func.args:
        if md[arg]['sql']:
            sql_ops.add(arg)

    for op in func.ops:
        if op.opcode == "kernel":
            opname, args = op.args[0], op.args[1:]
            if all(arg in sql_ops for arg in args):
                sql_ops.add(op)

    return sql_ops


def sql_roots(func, env):
    """
    Determine SQL 'root' ops, those are ops along fusion boundaries.
    """
    sql_roots = set()
    sql_ops = partition_sql(func, env)

    for op in func.ops:
        if op.opcode == "kernel" and op in sql_ops:
            if not all(use in sql_ops for use in func.uses[op]):
                sql_roots.add(op)


#------------------------------------------------------------------------
# Handlers
#------------------------------------------------------------------------

def op_kernel(interp, funcname, *args):
    op = interp.op

    function = op.metadata['kernel']
    overload = op.metadata['overload']

    py_func, signature = overload.func, overload.resolved_sig

    impl_overload = function.best_match(SQL, signature.argtypes)

    kernel = impl_overload.func
    sig    = impl_overload.resolved_sig
    assert sig == signature, (sig, signature)

    return kernel(*args)

def op_convert(interp, arg):
    raise TypeError("SQL type conversion not supported yet")

def op_ret(interp, arg):
    interp.halt()
    return arg

handlers = {
    'kernel':   op_kernel,
    'convert':  op_convert,
    'ret':      op_ret,
}
