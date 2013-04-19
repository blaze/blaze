"""
Build a ply lexer, but without the implicit magic and global
state, just load prebuilt ply parser and lexers at roughly the
same cost of loading a Python module.
"""

import functools
from ply.lex import Lexer, LexerReflect
from ply.yacc import ParserReflect, LRTable, LRParser

def memoize(obj):
    cache = obj.cache = {}
    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        if args not in cache:
            cache[args] = obj(*args, **kwargs)
        return cache[args]
    return memoizer

# lexer is passed as argument to ensure that memoization is
# unique for parser/lexer pair.

@memoize
def yaccfrom(module, tabmodule, lexer):
    # Get the module dictionary used for the parser
    _items = [(k,getattr(module,k)) for k in dir(module)]
    pdict = dict(_items)

    # Collect parser information from the dictionary
    pinfo = ParserReflect(pdict)
    pinfo.get_all()

    # Read the tables
    lr = LRTable()
    lr.read_table(tabmodule)

    lr.bind_callables(pinfo.pdict)
    return LRParser(lr,pinfo.error_func)

@memoize
def lexfrom(module, lexmodule):
    lexobj = Lexer()
    lexobj.lexoptimize = 1

    _items = [(k,getattr(module,k)) for k in dir(module)]
    ldict = dict(_items)

    # Collect parser information from the dictionary
    linfo = LexerReflect(ldict)
    linfo.get_all()

    lexobj.readtab(lexmodule,ldict)
    return lexobj
