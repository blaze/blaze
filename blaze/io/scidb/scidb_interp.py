"""
SciDB query generation and execution from Blaze AIR.
"""

from __future__ import absolute_import, division, print_function


import blaze
from blaze.io.scidb import AFL
from blaze.compute.air import interp

from .error import InterfaceError
from .query import execute_query, temp_name, Query
from .datadescriptor import SciDB_DDesc


#------------------------------------------------------------------------
# Interpreter
#------------------------------------------------------------------------

def compile(func, env):
    # TODO: we can assemble a query at compile time, but we can't abstract
    # over scidb array names. Not sure this makes sense...
    return func, env

def interpret(func, env, args, persist=False, **kwds):
    # TODO: allow mixing scidb and non-scidb data...

    dshape = func.type.restype
    descs = [arg.ddesc for arg in args]
    inputs = [desc.query for desc in descs]
    conns = [desc.conn for desc in descs]

    if len(set(conns)) > 1:
        raise InterfaceError(
            "Can only perform query over one scidb interface, got multiple")

    # Assemble query
    env = {'interp.handlers' : handlers}
    query = interp.run(func, env, None, args=inputs)
    [conn] = set(conns)

    code = []
    cleanup = []
    query.generate_code(code, cleanup, set())
    expr = query.result()

    result = _execute(conn, code, cleanup, expr, persist)
    return blaze.array(SciDB_DDesc(dshape, result, conn))


def _execute(conn, code, cleanup, expr, persist):
    if code:
        for stmt in code:
            execute_query(conn, stmt, persist=False)

    temp = temp_name()
    query = "store({expr}, {temp})".format(expr=expr, temp=temp)
    execute_query(conn, query, persist)

    if cleanup:
        for stmt in cleanup:
            execute_query(conn, stmt, persist=False)

    return Query(temp, args=(), kwds={})

#------------------------------------------------------------------------
# Handlers
#------------------------------------------------------------------------

def op_kernel(interp, funcname, *args):
    op = interp.op

    function = op.metadata['kernel']
    overload = op.metadata['overload']

    py_func, signature = overload.func, overload.resolved_sig

    impl_overload = function.best_match(AFL, signature.argtypes)

    kernel = impl_overload.func
    sig    = impl_overload.resolved_sig
    assert sig == signature, (sig, signature)

    return kernel(*args)

def op_convert(interp, arg):
    raise TypeError("scidb type conversion not supported yet")

def op_ret(interp, arg):
    interp.halt()
    return arg

handlers = {
    'kernel':   op_kernel,
    'convert':  op_convert,
    'ret':      op_ret,
}
