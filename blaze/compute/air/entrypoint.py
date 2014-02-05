# -*- coding: utf-8 -*-

"""
Assemble an execution kernel from a given expression graph.
"""

from __future__ import absolute_import, division, print_function
from . import pipeline, environment, passes, execution

def compile(expr, strategy, debug=False):
    """
    Prepare a Deferred for interpretation
    """
    env = environment.fresh_env(expr, strategy, debug=debug)
    if debug:
        passes_ = passes.debug_passes
    else:
        passes_ = passes.passes
    return pipeline.run_pipeline(expr, env, passes_)

def run(air_func, env, args, **kwds):
    """
    Prepare a Deferred for interpretation
    """
    # Find evaluator
    strategy = env['air.strategy']
    interp = execution.lookup_interp(strategy)

    return interp.interpret(air_func, env, args=args, **kwds)