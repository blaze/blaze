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

def valid_sql(op, env):
    """
    Determine whether `op` needs to be handled by the SQL backend.
    """
    sql_ops = env.setdefault('sql.ops', {})

    if isinstance(op, ir.FuncArg):
        # Function argument, this is a valid SQL query if the runtime input
        # described an SQL data source
        runtime_args = env['runtime.args']
        array = runtime_args[op]
        data_desc = array._data
        sql_ops[op] = data_desc.conn
        return True
    elif all(arg in sql_ops for arg in op.args[1:]):
        conn = sql_ops[op.args[1]]
        return all(conn == sql_ops[arg] for arg in op.args[1:])
    else:
        return False


determine_strategy = {
    'sql':      valid_sql,
    #'ooc':      valid_ooc,
    #'jit':      valid_jit,
    #'ckernel':  valid_ckernel,
    #'python':   valid_python,
}

#------------------------------------------------------------------------
# Partitioning
#------------------------------------------------------------------------

def partition(func, env):
    """
    Determine the execution strategy for each operation.
    """
    strategies = env['strategies'] = {}

    for arg in func.args:
        strategies[arg] = determine_preference(arg, env)

    for op in func.ops:
        if op.opcode == "kernel":
            strategies[arg] = determine_preference(arg, env)


def determine_preference(op, env):
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
