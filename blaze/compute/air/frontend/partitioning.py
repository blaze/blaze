"""
Plugin annotation and partitioning. This determines according to a set of
rules which plugin to use for which operation.
"""

from __future__ import absolute_import, division, print_function
from collections import defaultdict

import datashape

from ..ir import FuncArg
from ...strategy import CKERNEL
from ....io.sql import SQL, SQL_DDesc


# List of backends to use greedily listed in order of preference

preferences = [
    SQL,  # TODO: Allow easier extension of new backends
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

    if isinstance(op, FuncArg):
        # Function argument, this is a valid SQL query if the runtime input
        # described an SQL data source
        runtime_args = env['runtime.args']
        array = runtime_args[op]
        data_desc = array.ddesc
        is_scalar = not data_desc.dshape.shape
        if not isinstance(data_desc, SQL_DDesc) and not is_scalar:
            return False
        if isinstance(data_desc, SQL_DDesc):
            conns[op] = data_desc.conn
        return True
    elif all(strategies[arg] == SQL for arg in op.args[1:]):
        connections = set(conns[arg] for arg in op.args[1:] if arg in conns)
        if len(connections) == 1:
            [conn] = connections
            conns[op] = conn
            return True
        return False
    else:
        return False


determine_plugin = {
    SQL:        use_sql,
}

#------------------------------------------------------------------------
# Annotation
#------------------------------------------------------------------------

def annotate_all_kernels(func, env):
    """
    Annotate all sub-expressions with all kernels that can potentially
    execute the operation.

    Populate environment with 'kernel.overloads':

        { (Op, pluginname) : Overload }
    """
    # { (Op, pluginname) : Overload }
    impls = env['kernel.overloads'] = {}

    # { op: [plugin] }
    unmatched = env['unmached_impls'] = defaultdict(list)

    for op in func.ops:
        if op.opcode == "kernel":
            _find_impls(op, unmatched, impls)


def _find_impls(op, unmatched, impls):
    function = op.metadata['kernel']
    overload = op.metadata['overload']

    found_impl = False
    for pluginname in function.available_plugins:
        py_func, signature = overload_for_plugin(function, overload, pluginname)
        if py_func is not None:
            impls[op, pluginname] = py_func, signature
            found_impl = True
        else:
            unmatched[op].append(pluginname)


def overload_for_plugin(function, overload, pluginname):
    """Find an implementation overload for the given plugin"""
    expected_signature = overload.resolved_sig
    argstype = datashape.coretypes.Tuple(expected_signature.argtypes)

    try:
        overloader, datalist = function.plugins[pluginname]
        idx, match = overloader.resolve_overload(argstype)
    except datashape.OverloadError:
        return None, None

    if match != expected_signature and False:
        ckdispatcher = function.get_dispatcher('ckernel')
        raise TypeError(
            "Signature of implementation (%s) does not align with "
            "signature from blaze function (%s) from argtypes [%s] "
            "for function %s with signature %s" %
                (match, expected_signature,
                 ", ".join(map(str, expected_signature.argtypes)),
                 function, overload.sig))

    return datalist[idx], match

#------------------------------------------------------------------------
# Partitioning
#------------------------------------------------------------------------

def partition(func, env):
    """
    Determine the execution plugin for each operation.
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
    """Return the first valid plugin according to a list of preferences"""
    strategies = env['strategies']
    for preference in preferences:
        valid_plugin = determine_plugin[preference]
        if valid_plugin(op, strategies, env):
            return preference

    # If no alternative plugin was found, use the default ckernel
    return CKERNEL

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
