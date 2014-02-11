# -*- coding: utf-8 -*-

"""
Assemble an execution kernel from a given expression graph.
"""

from __future__ import absolute_import, division, print_function
from . import pipeline, environment, passes, execution

def compile(expr, storage):
    """
    Prepare a Deferred for interpretation
    """
    env = environment.fresh_env(expr, storage)
    return pipeline.run_pipeline(expr, env, passes.passes)

def run(air_func, env, args, **kwds):
    """
    Prepare a Deferred for interpretation
    """
    return execution.interpret(air_func, env, args=args, **kwds)