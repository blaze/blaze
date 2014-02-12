"""
Backend annotation and partitioning. This determines according to a set of
rules which backend to use for which operation.
"""

from __future__ import absolute_import, division, print_function
from pykit import ir
from ...strategy import OOC, JIT, CKERNEL, PY
from ....io.sql import SQL, SQLDataDescriptor


# List of backends to use greedily listed in order of preference

preferences = [
    SQL, # TODO: Allow easier extension of new backends
    #OOC,
    JIT,
    CKERNEL,
    PY,
]

#------------------------------------------------------------------------
# Strategies
#------------------------------------------------------------------------

# TODO: We may want the first N passes to have access to accurate types
#       (containing concrete shape) and runtime inputs, and then do our
#       instruction selection. After that we can throw this away to perform
#       caching from that point on.
#
#       Alternatively, we can encode everything as metadata early on, but this
#       may not work well for open-ended extension, such as plugins that were
#       not foreseen

def use_sql(op, strategies, env):
    """
    Determine whether `op` needs to be handled by the SQL backend.

    NOTE: This also populates env['sql.conns']. Mutating this way is somewhat
          undesirable, but this is a non-local decision anyway
    """
    conns = env.setdefault('sql.conns', {})

    if isinstance(op, ir.FuncArg):
        # Function argument, this is a valid SQL query if the runtime input
        # described an SQL data source
        runtime_args = env['runtime.args']
        array = runtime_args[op]
        data_desc = array._data
        if not isinstance(data_desc, SQLDataDescriptor):
            return False
        conns[op] = data_desc.conn
        return True
    elif all(strategies[arg] == SQL for arg in op.args[1:]):
        conn = conns[op.args[1]]
        return all(conn == conns[arg] for arg in op.args[1:])
    else:
        return False


def use_ooc(op, strategies, env):
    """
    Determine whether `op` needs to be handled by an out-of-core backend.
    """
    if isinstance(op, ir.FuncArg):
        # Function argument, this is an OOC operation if he runtime input
        # is persistent
        runtime_args = env['runtime.args']
        array = runtime_args[op]
        data_desc = array._data
        return data_desc.capabilities.persistent

    ooc = all(strategies[arg] in (PY, CKERNEL, JIT, OOC) for arg in op.args[1:])
    return ooc and not use_local(op, strategies, env)


def use_local(op, strategies, env):
    """
    Determine whether `op` can be handled by a 'local' backend.
    """
    if isinstance(op, ir.FuncArg):
        # Function argument, this is an OOC operation if he runtime input
        # is persistent
        runtime_args = env['runtime.args']
        array = runtime_args[op]
        data_desc = array._data
        return True # not data_desc.capabilities.remote
        #return not (data_desc.capabilities.persistent or
        #            data_desc.capabilities.remote)

    return all(strategies[arg] in local_strategies for arg in op.args[1:])


local_strategies = (JIT, CKERNEL, PY)

determine_strategy = {
    SQL:        use_sql,
    #OOC:        use_ooc,
    JIT:        use_local,
    CKERNEL:    use_local,
    PY:         use_local,
}

#------------------------------------------------------------------------
# Annotation
#------------------------------------------------------------------------

def annotate_all_kernels(func, env):
    """
    Annotate all sub-expressions with all kernels that can potentially
    execute the operation.

    Populate environment with 'kernel.overloads':

        { (Op, strategy) : Overload }
    """
    impls = env['kernel.overloads'] = {} # { (Op, strategy) : Overload }

    for op in func.ops:
        if op.opcode == "kernel":
            _find_impls(op, env, impls)


def _find_impls(op, env, impls):
    function = op.metadata['kernel']
    overload = op.metadata['overload']

    found_impl = False
    for strategy in enumerate_strategies(function, env):
        result = overload_for_strategy(function, overload, strategy)
        if result:
            impls[op, strategy] = result
            found_impl = True

    if not found_impl:
        raise TypeError("No implementation found for %s" % (function,))


def enumerate_strategies(function, env):
    """Return the available strategies for the blaze function for this op"""
    return function.available_strategies


def overload_for_strategy(function, overload, strategy):
    """Find an implementation overload for the given strategy"""
    expected_signature = overload.resolved_sig
    argtypes = expected_signature.argtypes

    if function.matches(strategy, argtypes):
        overload = function.best_match(strategy, argtypes)
        got_signature = overload.resolved_sig

        # Assert agreeable types for now
        # TODO: insert conversions if implementation disagrees

        if got_signature != expected_signature and False:
            ckdispatcher = function.get_dispatcher('ckernel')
            raise TypeError(
                "Signature of implementation (%s) does not align with "
                "signature from blaze function (%s) from argtypes [%s] "
                "for function %s with signature %s" %
                    (got_signature, expected_signature,
                     ", ".join(map(str, argtypes)),
                     function, overload.sig))

        return overload.func, got_signature

    return None, None


#def annotate_kernels(func, env):
#    """
#    Annotate each op with an implementation.
#    """
#    overloads = env['kernel.overloads']
#    strategies = env['strategies']
#    impls = env['kernel.impls'] = {}
#
#    for op in func.ops:
#        if op.opcode == 'kernel':
#            strategy = strategies[op]
#            overload = overloads[op, strategy]
#            impls[op] = overload

#------------------------------------------------------------------------
# Partitioning
#------------------------------------------------------------------------

def partition(func, env):
    """
    Determine the execution strategy for each operation.
    """
    strategies = env['strategies'] = {}
    impls = env['kernel.overloads']

    for arg in func.args:
        strategies[arg] = determine_preference(arg, env, preferences)

    for op in func.ops:
        if op.opcode == "kernel":
            prefs = [p for p in preferences if (op, p) in impls]
            strategies[op] = determine_preference(op, env, prefs)


def determine_preference(op, env, preferences):
    """Return the first valid strategy according to a list of preferences"""
    strategies = env['strategies']
    for preference in preferences:
        valid_strategy = determine_strategy[preference]
        if valid_strategy(op, strategies, env):
            return preference

    #import pdb; pdb.set_trace()
    raise ValueError(
        "No valid strategy could be determined for %s from %s" % (op, preferences))

#------------------------------------------------------------------------
# Backend boundaries / Fusion boundaries
#------------------------------------------------------------------------

def annotate_roots(func, env):
    """
    Determine 'root' ops, those are ops along fusion boundaries. E.g.
    a unary kernel that can only operate on in-memory data, with an sql
    operand expression:

        kernel(expr{sql}){jit}

    Roots become the place where some backend-specific (e.g. 'jit', or 'sql')
    must return some blaze result to the execution engine, that describes the
    data (e.g. via an out-of-core, remote or local data descriptor).

    NOTE: The number and nature of uses of a root can govern where and if
          to move the data. For instance, if we do two local computations on
          one remote data source, we may want to move it just once first (
          or in chunks)
    """
    strategies = env['strategies']
    roots = env['roots'] = set()

    for op in func.ops:
        if op.opcode == "kernel":
            uses = func.uses[op]
            if len(uses) > 1:
                # Multiple uses, boundary
                roots.add(op)
            elif any(strategies[arg] != strategies[op] for arg in op.args[1:]):
                # Different exeuction strategies, boundary
                roots.add(op)
            elif len(uses) == 1:
                # Result for user, boundary
                [use] = uses
                if use.opcode == 'ret':
                    roots.add(op)
