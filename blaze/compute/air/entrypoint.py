"""
Assemble an execution kernel from a given expression graph.
"""

from __future__ import absolute_import, division, print_function

from . import pipeline, environment, passes, execution

def compile(expr, ddesc, debug=False):
    """
    Prepare a Deferred for interpretation
    """
    env = environment.fresh_env(expr, ddesc, debug=debug)
    if debug:
        passes_ = passes.debug_passes
    else:
        passes_ = passes.passes
    return pipeline.run_pipeline(expr, env, passes_)


def run(air_func, env, **kwds):
    """
    Prepare a Deferred for interpretation
    """
    return execution.interpret(air_func, env, **kwds)
