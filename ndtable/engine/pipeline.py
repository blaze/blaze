"""
Defines the Pipeline class which provides a series of transformation
passes on the graph which result in code generation.

This is fundamentally quite different from the Numba pipeline in that we
start with a graph object and start generating LLVM immedietely. At some
point the two might merge though if we're sufficiently clever...
"""

from llvm.core import Module
from functools import partial
from collections import Counter, defaultdict

from ndtable.datashape.coretypes import _var_generator

#------------------------------------------------------------------------
# Pipeline Combinators
#------------------------------------------------------------------------

def compose(f, g):
    return lambda *x: f(*g(*x))

#------------------------------------------------------------------------
# Uid Generator
#------------------------------------------------------------------------

# Generate a stream of unique identifiers in the context of the
# pipeline.
#   func1, func2...
#   sym1, sym2...

def uids(prefix):
    return partial(_var_generator, prefix)

#------------------------------------------------------------------------
# Passes
#------------------------------------------------------------------------

#             Input
#               |
#               |
# +----------------------+
# |          pass 1      |
# +--------|----------|--+
#        context    graph
#          |          |
# +--------|----------|--+
# |          pass 2      |
# +--------|----------|--+
#        context    graph
#          |          |
# +--------|----------|--+
# |          pass 3      |
# +--------|----------|--+
#          |          |
#          +----------+-----> Output

def do_flow(context, graph):
    context = dict(context)

    # Topologically sort the graph
    sort = toposort(graph)

    # ----------------------
    context['order'] = sort
    # ----------------------

    return context, graph

def do_environment(context, graph):
    context = dict(context)

    # ----------------------
    context['llvmmodule'] = Module.new('blaze')
    # ----------------------

    return context, graph

def do_codegen(context, graph):
    context = dict(context)

    return context, graph

def do_plan(context, graph):
    context = dict(context)

    # ----------------------
    context['output'] = None
    # ----------------------

    return context, graph

#------------------------------------------------------------------------
# Pipeline
#------------------------------------------------------------------------

# TODO: there is no reason this should be a class, and classes
# just complicate things and encourage nasty things like
# inheritance...

class Pipeline(object):
    """
    Code generation pipeline is a series of combinable Pass
    stages which thread a context and graph object through to
    produce various intermediate forms resulting in an execution
    plan.
    """
    def __init__(self, *args, **kwargs):
        self.ictx = {}

        # sequential pipeline of passes
        self.pipeline = (
            # Input <--
            do_flow,
            #  v
            do_environment,
            #  v
            do_codegen,
            # Output -->
            do_plan,
            # Output -->
        )

    def run_pipeline(self, graph):
        """
        Run the graph through the pipeline, spit out a LLVM
        module.
        """
        ictx = self.ictx

        # Fuse the passes into one functional pipeline that is the
        # sequential composition with the intermediate ``context`` and
        # ``graph`` objects threaded through.
        pipeline = reduce(compose, self.pipeline)

        octx, _ = pipeline(ictx, graph)
        return octx['output']

#------------------------------------------------------------------------
# Graph Manipulation
#------------------------------------------------------------------------

def toposort(graph):
    """
    Sort the expression graph topologically to resolve the order needed
    to execute operations.

         +
        / \
       a   +     --> [a, b, c]
          / \
         b   c
             |
             a

    """

    result = []
    count = Counter()

    for node in iter(graph):
        for child in iter(node):
            count[child] += 1

    sort = [node for node in graph if not count[node]]

    while sort:
        node = sort.pop()
        result.append(node)

        for child in iter(node):
            count[child] -= 1
            if count[child] == 0:
                sort.append(child)

    result.reverse()
    return result
