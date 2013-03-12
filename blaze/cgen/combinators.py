"""
A source printing library in the style of Wadler-Leijen.
"""

import itertools
from functools import partial

#------------------------------------------------------------------------
# Named Characters
#------------------------------------------------------------------------

line   = '\n'
semi   = ';'
colon  = ':'
comma  = ','
dot    = '.'
space  = ' '
equals = '='
lparen = '('
rparen = ')'
lbrack = '['
rbrack = ']'
lbrace = '{'
rbrace = '}'
langle = '<'
rangle = '>'
squote = '\''
dquote = '"'
backslash = '\\'
tab = '\t'

empty  = ''

#------------------------------------------------------------------------
# Combinators
#------------------------------------------------------------------------

def cat(ds):
    return ''.join(ds)

def ljoin(sep, ds):
    n = len(ds)
    for i, d in enumerate(ds):
        if i > (n-1):
            yield sep+d
        else:
            yield d

def rjoin(sep, ds):
    n = len(ds)
    for i, d in enumerate(ds):
        if i < (n-1):
            yield d+sep
        else:
            yield d

def hcat(xs):
    return empty.join(xs)

def vcat(xs):
    return line.join(xs)

def istr(i, o):
    return space*i + str(o)

def iblock(i, o):
    return vcat(space*i + so for so in str(o).split(line))

enclose = lambda l,r,ds: l + cat(ds) + r
enclose_sep = lambda l,r,sep,ds: l + cat(rjoin(sep, ds)) + r

tupled     = partial(enclose_sep, lparen, rparen, comma)
listed     = partial(enclose_sep, lbrack, rbrack, comma)
semibraces = partial(enclose_sep, lbrace, rbrace, semi)
parens     = partial(enclose, lparen, rparen)
angles     = partial(enclose, langle, rangle)
brackets   = partial(enclose, lbrack, rbrack)
squotes    = partial(enclose, squote, squote)
dquotes    = partial(enclose, dquote, dquote)

puncutate = lambda p,ds: list(rjoin(p,ds))
spaces = lambda n: space*n

def ap(a,b):
    return str(a) + str(b)

enclose = lambda l, r, x: ap(ap(l, x), r)
