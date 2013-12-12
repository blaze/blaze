# -*- coding: utf-8 -*-

"""
SciDB query generation and execution. The query building themselves are
fairly low-level, since their only concern is whether to generate temporary
arrays or not.
"""

from __future__ import print_function, division, absolute_import
import uuid
from itertools import chain

#------------------------------------------------------------------------
# Query Interface
#------------------------------------------------------------------------

def temp_name():
    return 'arr_' + str(uuid.uuid4()).replace("-", "_")

class Query(object):
    """
    Holds an intermediate SciDB query. This builds up a little query graph
    to deal with expression reuse.

    For instance, consider the code:

        b = a * 2
        eval(b + b)

    This would generate the query "(a * 2) + (a * 2)". In this case the
    blaze expression graph itself knows about the duplication of the
    expression. However, scidb kernels may themselves reuse expressions
    multiple times, which can lead to exponential code generation.

    E.g. consider function `f(a) = a * a`. Now f(f(f(a))) has `a` 8 times.
    """

    temp_name = None

    def __init__(self, pattern, args, kwds, interpolate=str.format):
        self.pattern = pattern
        self.args = args
        self.kwds = kwds
        self.interpolate = interpolate
        self.uses = []

        for arg in chain(self.args, self.kwds.values()):
            if isinstance(arg, Query):
                arg.uses.append(self)

    def _result(self):
        """
        Format the expression.
        """
        return self.interpolate(self.pattern, *self.args, **self.kwds)

    def generate_code(self, code, cleanup, seen):
        """
        Generate a query to produce a temporary array for the expression.
        The temporary array can be referenced multiple times.
        """
        if self in seen:
            return
        seen.add(self)

        for arg in chain(self.args, self.kwds.values()):
            if isinstance(arg, Query):
                arg.generate_code(code, cleanup, seen)

        if len(self.uses) > 1:
            self.temp_name = temp_name()
            code.append("store({expr}, {temp})".format(expr=self._result(),
                                                       temp=self.temp_name))
            cleanup.append("remove({temp})".format(temp=self.temp_name))

    def result(self):
        """
        The result in the AFL expression we are building.
        """
        if len(self.uses) > 1:
            return self.temp_name
        return self._result()

    def __str__(self):
        if self.temp_name:
            return self.temp_name
        return self.result()


def qformat(s, *args, **kwds):
    return Query(s, args, kwds)

#------------------------------------------------------------------------
# Query Execution
#------------------------------------------------------------------------

def execute_query(conn, query, persist=False):
    return conn.query(query, persist=persist)

#------------------------------------------------------------------------
# Query Generation
#------------------------------------------------------------------------

def apply(name, *args):
    arglist = ["{%d}" % (i,) for i in range(len(args))]
    pattern = "{name}({arglist})".format(name=name, arglist=", ".join(arglist))
    return qformat(pattern, *args)

def expr(e):
    return qformat("({0})", expr)

def iff(expr, a, b):
    return apply("iff", expr, a, b)

def build(arr, expr):
    return apply("build", arr, expr)