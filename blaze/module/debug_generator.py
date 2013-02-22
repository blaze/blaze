#!/usr/bin/env python

import sys
import readline
import argparse
from pprint import pprint
from parser import mopen

from functools import partial
import debug_passes as debug

banner = """mod-debug
------------------------------------
Type :help for for more information.
"""

help = """
-- Usage:

-- Commmands:

  :show
  :type
  :let
  :load
  :browse
  :help

"""

def completer(mod, text, state):
    opts = []
    #opts = [i for i in mod.keys() if i.startswith(text)]
    if state < len(opts):
        return opts[state]
    else:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('module', nargs='?', help='Module')
    parser.add_argument('--noprelude', action='store_true', help='Include prelude')
    parser.add_argument('--ddump-mod', action='store_true', help='Dump module tree')
    args = parser.parse_args()

    mod = None

    if args.module:
        mod = mopen(args.module)
    else:
        print 'No module specified?'
        sys.exit(0)

    if not args.noprelude:
        pass

    readline.parse_and_bind("tab: complete")
    readline.set_completer(partial(completer, mod))

    while True:
        try:
            line = raw_input('>> ').strip()
        except EOFError:
            break

        #-----------------------------------------------
        if line.startswith(':show') or line.startswith(':s'):
            try:
                rr = mod[line[1:].strip()]
                print rr
            except KeyError:
                print "No such rule or strategy '%s'" % line[1:]

        #-----------------------------------------------
        elif line.startswith(':type') or line.startswith(':t'):
            pass

        elif line.startswith(':reload') or line.startswith(':r'):
            print "Reloading".center(80, '=')
            mod = mopen(args.module)

        #-----------------------------------------------
        elif line.startswith(':let'):
            pass

        #-----------------------------------------------
        elif line.startswith(':load'):
            pass

        #-----------------------------------------------
        elif line.startswith(':browse'):
            pass

        #-----------------------------------------------
        elif line.startswith(':help'):
            print help
            pass

        #-----------------------------------------------
        elif line.startswith(':bound'):
            head = line.split()[1]
            cls = mod.resolve_bound(head)

            for name, sig in cls:
                print '\t', name, '::', sig

        elif line.startswith(':adhoc'):
            fn = line.split()[1]

            try:
                cls = mod.anon_refs[fn]

                for name, sig in cls:
                    print '\t', name, '::', sig
            except KeyError:
                print 'No definition'

        else:
            pass

if __name__ == '__main__':
    main()
