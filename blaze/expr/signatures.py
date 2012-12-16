"""
Convert between type signatures between Blaze and NumPy

::
    NumPy: ii -> i
    Blaze: (a,a) -> a

"""

import re
from blaze.expr.typeinference import tyeval
from blaze.datashape import shorthand as T

# TODO: if we extend this then just use ply
sepr = re.compile(r'(\w+(?=\))|\w)')
mapr = re.compile(r"(.*)->(.*)")
whitespace = re.compile(r'\s')

class SVar(object):
    def __init__(self, var):
        self.var = var
    def __repr__(self):
        return 'SVar(%s)' % self.var

#------------------------------------------------------------------------
# Toplevel
#------------------------------------------------------------------------

def sig_parse(s):
    """
    Parameters
    ----------
    s : str
        signature string to be parsed

    Returns
    -------
    out : tuple
        outputted variable strings

    Usage
    -----
    >>> parse('(a,a) -> a')
    ([SVar(a), SVar(a)], [SVar(a)])
    >>> parse('ii->i')
    ([SVar(i), SVar(i)], [SVar(y)])
    """
    # strip redundent whitespace
    s = whitespace.sub('', s)
    tok = mapr.split(s)

    dom = sepr.findall(tok[1])
    cod = sepr.findall(tok[2])

    #return (SVar, dom), map(SVar, cod)
    return dom, cod
