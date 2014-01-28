# -*- coding: utf-8 -*-

"""
AIR compilation environment.
"""

from __future__ import print_function, division, absolute_import

# Any state that should persist between passes end up in the environment, and
# should be documented here

air_env = {
    # blaze expression graph
    #'expr_graph':       None,

    # execution strategy
    'strategy':         None,
    # strategy determined for each Op: { Op : strategy }
    # For instance different sub-expressions may be execution in different
    # environments
    'strategies':       None,
    # Runtime input arguments
    'runtime.args':     None,
}

def fresh_env(expr, strategy):
    """
    Allocate a new environment.
    """
    env = dict(air_env)
    env['strategy'] = strategy
    return env
