# -*- coding: utf-8 -*-

"""
Blaze types interpreter.
"""

import os
import sys
import code
import atexit
import logging
import readline
import warnings
import rlcompleter

import blaze
from blaze import dshape, dshapes
from blaze.datashape import (unify_simple as unify,
                             normalize_ellipses as normalize,
                             promote, tmap, coercion_cost, typeof)

logging.getLogger('blaze').setLevel(logging.DEBUG)

banner = """
The Blaze typing interpreter.

    blaze:
        blaze module

    dshape('<type string>'):
        parse a blaze type

    dshapes('<type string1>', ..., '<type string N>')
        parse a series of blaze types in the same context, so they will
        shared type variables of equal name.

    typeof(val)
        Return a blaze DataShape for a python object

    unify(t1, t2):
        unify t1 with t2, and return a result type and a list of additional
        constraints

    promote(t1, t2):
        promote two blaze types to a common type general enough to represent
        values of either type

    normalize(ds1, ds2):
        normalize two datashapes for unification (ellipses, broadcasting)

    coercion_cost(t1, t2):
        Determine a coercion cost for coercing type t1 to type t2

    tmap(f, t):
        map function `f` over type `t` and its sub-terms post-order
"""

env = {
    'blaze':     blaze,
    'dshape':    dshape,
    'dshapes':   dshapes,
    'typeof':    typeof,
    'unify':     unify,
    'promote':   promote,
    'normalize': normalize,
    'coercion_cost': coercion_cost,
    'tmap':      tmap,
}

def init_readline():
    readline.parse_and_bind('tab: menu-complete')
    histfile = os.path.expanduser('~/.blaze_history%s' % sys.version[:3])
    atexit.register(readline.write_history_file, histfile)
    if not os.path.exists(histfile):
        open(histfile, 'w').close()
    readline.read_history_file(histfile)

def main():
    init_readline()
    try:
        import fancycompleter
        print(banner)
        fancycompleter.interact(persist_history=True)
    except ImportError:
        warnings.warn("fancycompleter not installed")
        interp = code.InteractiveConsole(env)
        interp.interact(banner)

if __name__ == '__main__':
    main()
