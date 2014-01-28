"""
Backend annotation and partitioning. This determines according to a set of
rules which backend to use for which operation.
"""

from __future__ import absolute_import, division, print_function
from pykit import ir

# List of backends to use greedily listed in order of preference

preferences = [
    'sql',
    'jit',
    'ckernel',
    'python',
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
        conns[op] = data_desc.conn
        return True
    elif all(strategies[arg] == 'sql' for arg in op.args[1:]):
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
        return data_desc.capabilities['persistent']

    ooc = all(strategies[arg] in ('local', 'ooc') for arg in op.args[1:])
    return ooc and not use_local(op,  strategies, env)


def use_local(op, strategies, env):
    """
    Determine whether `op` needs to be handled by an out-of-core backend.
    """
    if isinstance(op, ir.FuncArg):
        # Function argument, this is an OOC operation if he runtime input
        # is persistent
        runtime_args = env['runtime.args']
        array = runtime_args[op]
        data_desc = array._data
        return not (data_desc.capabilities['persistent'] or
                    data_desc.capabilities['remote'])

    return all(strategies[arg] in ('local', 'ooc') for arg in op.args[1:])


determine_strategy = {
    'sql':      use_sql,
    'ooc':      use_ooc,
    'local':    use_local,
}

#------------------------------------------------------------------------
# Partitioning
#------------------------------------------------------------------------

def annotate_kernels(func, env):
    """
    Annotate all sub-expressions with all kernels that can potentially
    execute the operation.
    """
    impls = env['kernel.impls'] = {} # { (Op, strategy) : Overload }

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

        assert got_signature == expected_signature, (got_signature,
                                                     expected_signature)

        return overload.func, got_signature

    return None, None

#------------------------------------------------------------------------
# Partitioning
#------------------------------------------------------------------------

def partition(func, env):
    """
    Determine the execution environment strategy for each operation.
    """
    strategies = env['exc_strategies'] = {}
    impls = env['kernel.impls']

    for arg in func.args:
        strategies[arg] = determine_preference(arg, strategies, preferences)

    for op in func.ops:
        if op.opcode == "kernel":
            prefs = [p for p in preferences if (op, p) in impls]
            strategies[op] = determine_preference(op, strategies, prefs)


def partition_local(func, env):
    """
    Determine the local execution backend strategy for each operation.
    """
    exc_strategies = env['exc_strategies'] = {}
    strategies = env.setdefault('strategies', {})

    for op in func.ops:
        if op.opcode == "kernel":
            strategies[op] = determine_preference(op, strategies)


def determine_preference(op, env, preferences):
    for preference in preferences:
        valid_strategy = determine_strategy[preference]
        if valid_strategy(op, env):
            return preference

    raise ValueError("No valid strategy could be determined for %s" % (op,))


def annotate_roots(func, env):
    """
    Determine 'root' ops, those are ops along fusion boundaries.
    """
    strategies = env['strategies']
    roots = env['roots'] = set()

    for op in func.ops:
        if op.opcode == "kernel":
            if func[op.uses] > 1:
                roots.add(op)
            elif any(strategies[arg] != strategies[op] for arg in op.args[1:]):
                roots.add(op)
