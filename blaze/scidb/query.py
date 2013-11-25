# -*- coding: utf-8 -*-

"""
SciDB query generation and execution. The query building themselves are
fairly low-level, since their only concern is whether to generate temporary
arrays or not.
"""

from __future__ import print_function, division, absolute_import

#------------------------------------------------------------------------
# Query Interface
#------------------------------------------------------------------------

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

    def __init__(self, pattern, args, interpolate=str.format):
        self.pattern = pattern
        self.args = args
        self.interpolate = interpolate
        self.uses = []

    def generate_code(self, code, cleanup):
        """
        Generate a query to produce a temporary array for the expression.
        The temporary array can be referenced multiple times.
        """

    def result(self):
        """
        The result in the AFL expression we are building.
        """
        return self.interpolate(self.pattern, self.args)

    def __str__(self):
        return self.result()


class QueryTemp(Query):
    """
    Query that assigns itself to a temporary array for multiple use. In an
    AFL expression, this constitutes the temporary array name.
    """

    def generate_code(self, code, cleanup):
        # TODO: implement
        code.emit(self.expr)
        self.temp_name = "my_temp_array"

    def result(self):
        return self.temp_name


def qformat(s, *args):
    return Query(s, args)

#------------------------------------------------------------------------
# Query Execution
#------------------------------------------------------------------------

def execute_query(interface, query, persist=False):
    raise NotImplementedError

#------------------------------------------------------------------------
# Query Generation
#------------------------------------------------------------------------

def apply(name, *args):
    arglist = ["{%d}" % (i,) for i in range(len(args))]
    pattern = "{name}({arglist})".format(name=name, arglist=arglist)
    return qformat(pattern, *args)

def expr(e):
    return qformat("({0})", expr)

def iff(expr, a, b):
    return apply("iff", expr, a, b)

def build(arr, expr):
    return apply("build", arr, expr)