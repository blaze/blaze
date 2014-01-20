# -*- coding: utf-8 -*-

"""
AIR compilation environment.
"""

from __future__ import print_function, division, absolute_import

# Any state that should persist between passes end up in the environment, and
# should be documented here

air_env = {
    #'air.expr_graph':       None,   # blaze expression graph
    'air.strategy':         None,   # execution strategy
}

def fresh_env(expr, strategy):
    """
    Allocate a new environment.
    """
    env = dict(air_env)
    env['air.strategy'] = strategy
    return env
