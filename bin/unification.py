# -*- coding: utf-8 -*-

"""
Blaze types interpreter.
"""

import code
import readline # don't kill this import

import blaze
from blaze.datashape import unify, unify_simple, promote

banner = """
The Blaze typing interpreter.

    blaze:
        blaze module

    dshape('<type string>'):
        parse a blaze type

    unify(t1, t2):
        unify t1 with t2, and return a result type and a list of additional
        constraints

    promote(t1, t2):
        promote two blaze types to a common type general enough to represent
        values of either type
"""

env = {
    'blaze': blaze,
    'dshape': blaze.dshape,
    'unify': unify_simple,
    'promote': promote,
}

def main():
    interp = code.InteractiveConsole(env)
    interp.interact(banner)

if __name__ == '__main__':
    main()