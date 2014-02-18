# -*- coding: utf-8 -*-

"""
Rewrite SQL operations in AIR. Generate SQL queries and execute them at roots.
"""

from __future__ import absolute_import, division, print_function

from . import db, SQL

from ... import Array
from .error import SQLError
from .query import execute, dynd_chunk_iterator
from .syntax import reorder_select, emit, Table, Column
from .datadescriptor import SQLDataDescriptor, SQLResultDataDescriptor
from ...datadescriptor import DyNDDataDescriptor

from pykit.ir import Op

def rewrite_sql(func, env):
    """
    Generate SQL queries for each SQL op and assemble them into one big query
    which we rewrite to python kernels.
    """
    strategies = env['strategies']      # op -> strategy (e.g. 'sql')
    impls = env['kernel.overloads']     # (op, strategy) -> Overload
    roots = env['roots']                # Backend boundaries: { Op }
    args = env['runtime.args']          # FuncArg -> blaze.Array
    conns = env['sql.conns']            # Op -> SQL Connection

    rewrite = set()                     # ops to rewrite to sql kernels
    delete  = set()                     # ops to delete
    queries = {}                        # op -> query (str)

    leafs = {}                          # op -> set of SQL leafs

    # Extract table names and insert in queries
    for arg in func.args:
        if strategies[arg] == 'sql':
            arr = args[arg]
            sql_ddesc = arr._data

            if isinstance(sql_ddesc, DyNDDataDescriptor):
                # Extract scalar value from blaze array
                assert not sql_ddesc.dshape.shape
                # Do something better here
                query = str(sql_ddesc.dynd_arr())
            else:
                table = Table(sql_ddesc.col.table)
                query = Column(table, sql_ddesc.col.colname)

            queries[arg] = query
            leafs[arg] = [arg]

    # print(func)
    # print(strategies)

    # Generate SQL queries for each op
    for op in func.ops:
        if op.opcode == "kernel" and strategies[op] == 'sql':
            query_gen, signature = impls[op, 'sql']

            args = op.args[1:]
            inputs = [queries[arg] for arg in args]
            query = query_gen(*inputs)
            queries[op] = query
            if args[0] in conns:
                conns[op] = conns[args[0]]
            leafs[op] = [leaf for arg in args
                                  for leaf in leafs[arg]]

        elif op.opcode == 'convert':
            uses = func.uses[op]
            if all(strategies[use] == 'sql' for use in uses):
                arg = op.args[0]
                query = queries[arg]
                queries[op] = query

                if arg in conns:
                    conns[op] = conns[arg]
                leafs[op] = list(leafs[arg])
            else:
                continue

        else:
            continue

        if op in roots:
            rewrite.add(op)
        else:
            delete.add(op)

    # Rewrite sql kernels to python kernels
    for op in rewrite:
        query = queries[op]
        pykernel = sql_to_pykernel(query, op, env)
        newop = Op('pykernel', op.type, [pykernel, leafs[op]], op.result)
        op.replace(newop)

    # Delete remaining unnecessary ops
    func.delete_all(delete)


def sql_to_pykernel(expr, op, env):
    """
    Create an executable pykernel that executes the given query expression.
    """
    conns = env['sql.conns']
    conn = conns[op]
    dshape = op.type

    query = reorder_select(expr)
    select_query = emit(query)

    def sql_pykernel(*inputs):
        try:
            # print("executing...", select_query)
            result = execute(conn, dshape, select_query, [])
        except db.OperationalError as e:
            raise db.OperationalError(
                "Error executing %s: %s" % (select_query, e))

        return Array(SQLResultDataDescriptor(result))

    return sql_pykernel


def compose_sql_select_query(tables, joins, expr):
    """
    Compose a select query from the given expression.

        SELECT <expr>
        FROM <tables>
        WHERE <cond>
    """
    assert len(joins) >= len(tables) - 1

    if not joins:
        [table] = tables
        return """SELECT %s
                  FROM %s""" % (expr, table)

    return """
        SELECT %s
        FROM %s
        WHERE %s
    """ % (expr, ", ".join(map(str, tables)), ", ".join(map(str, joins)))