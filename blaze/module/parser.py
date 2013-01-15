"""
If the input shape and output shape of every operation is specified,
then when they are chained together in expressions it is possible
to statically reason about the manner in which the data will be
transformed.

Replace *all* the NumPy methods, constructors and interators with
graph nodes with signatures.
"""

import re
import os
import sys

from functools import partial
from collections import namedtuple
from pprint import pprint, pformat

try:
    import mlex
    import myacc
    DEBUG = False
except:
    DEBUG = True

from blaze.plyhacks import yaccfrom, lexfrom
from blaze.error import CustomSyntaxError

#------------------------------------------------------------------------
# Errors
#------------------------------------------------------------------------

syntax_error = """

  File {filename}, line {lineno}
    {line}
    {pointer}

{error}: {msg}
"""

class ModuleSyntaxError(Exception):
    """
    Makes datashape parse errors look like Python SyntaxError.
    """
    def __init__(self, lineno, col_offset, filename, text, msg=None):
        self.lineno     = lineno
        self.col_offset = col_offset
        self.filename   = filename
        self.text       = text
        self.msg        = msg or 'invalid syntax'

    def __str__(self):
        return syntax_error.format(
            filename = self.filename,
            lineno   = self.lineno,
            line     = self.text,
            pointer  = ' '*self.col_offset + '^',
            msg      = self.msg,
            error    = self.__class__.__name__,
        )

#------------------------------------------------------------------------
# Lexer
#------------------------------------------------------------------------

tokens = (
    'INTERFACE', 'NAME', 'ARROW', 'COLON', 'OP', 'FUN', 'DCOLON', 'PYOP',
    'COMMA'
)

infix = ( '+', '*', '-', '/')

ops = set([
      '__abs__'
    , '__add__'
    , '__and__'
    , '__call__'
    , '__cmp__'
    , '__complex__'
    , '__contains__'
    , '__del__'
    , '__delattr__'
    , '__delitem__'
    , '__divmod__'
    , '__div__'
    , '__divmod__'
    , '__enter__'
    , '__exit__'
    , '__eq__'
    , '__floordiv__'
    , '__float__'
    , '__ge__'
    , '__getattr__'
    , '__getattribute__'
    , '__getitem__'
    , '__gt__'
    , '__hex__'
    , '__iadd__'
    , '__iand__'
    , '__idiv__'
    , '__ifloordiv__'
    , '__ilshift__'
    , '__imod__'
    , '__imul__'
    , '__init__'
    , '__int__'
    , '__invert__'
    , '__ior__'
    , '__ipow__'
    , '__irshift__'
    , '__isub__'
    , '__iter__'
    , '__ixor__'
    , '__le__'
    , '__len__'
    , '__long__'
    , '__lshift__'
    , '__lt__'
    , '__mod__'
    , '__mul__'
    , '__ne__'
    , '__neg__'
    , '__new__'
    , '__nonzero__'
    , '__oct__'
    , '__or__'
    , '__pos__'
    , '__pow__'
    , '__radd__'
    , '__rand__'
    , '__rdiv__'
    , '__repr__'
    , '__reversed__'
    , '__rfloordiv__'
    , '__rlshift__'
    , '__rmod__'
    , '__rmul__'
    , '__ror__'
    , '__rpow__'
    , '__rrshift__'
    , '__rshift__'
    , '__rsub__'
    , '__rxor__'
    , '__setattr__'
    , '__setitem__'
    , '__str__'
    , '__sub__'
    , '__unicode__'
    , '__xor__'
])

literals = [
    '=' ,
    ',' ,
    '(' ,
    ')' ,
    '{' ,
    '}' ,
]

t_COMMA  = r','
t_ignore = '\x20\x09\x0A\x0D'

def t_INTERFACE(t):
    r'interface'
    return t

def t_PYOP(t):
    r'_\+_|_\*_|_-_|_i/_|_\*\*_|_\>_|_\<_'
    return t

def t_OP(t):
    r'op'
    return t

def t_FUN(t):
    r'fun'
    return t

def t_DCOLON(t):
    r'::'
    return t

def t_COLON(t):
    r'\:'
    return t

def t_ARROW(t):
    r'->'
    return t

def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")

def t_NAME(t):
    r'[a-zA-Z_][a-zA-Z0-9_\']*'
    return t

def t_COMMENT(t):
    r'\#.*'
    pass

def t_error(t):
    print("Unknown token '%s'" % t.value[0])
    t.lexer.skip(1)

#------------------------------------------------------------------------
# Parser
#------------------------------------------------------------------------

interface = namedtuple('interface', 'name params body')
opdef = namedtuple('opdef', 'op sig')
fndef = namedtuple('fndef', 'name sig')
sig = namedtuple('interface', 'dom cod')
fn = namedtuple('fn', 'dom cod')
pt = namedtuple('pt', 'x xs')

def p_top(p):
    '''top : mod'''
    p[0] = p[1]

#------------------------------------------------------------------------

def p_decl1(p):
    'mod : mod mod'
    p[0] = p[1] + p[2]

def p_decl2(p):
    'mod : stmt'
    p[0] = [p[1]]

#------------------------------------------------------------------------

def p_statement_assign(p):
    '''stmt : INTERFACE params COLON def
            | INTERFACE params COLON empty
    '''
    p[0] = interface(p[2][0],p[2][1:],p[4])

def p_params1(p):
    'params : params params'
    p[0] = p[1] + p[2]

def p_params2(p):
    'params : NAME '
    p[0] = (p[1],)

def p_def1(p):
    'def : def def'
    p[0] = p[1] + p[2]

def p_def2(p):
    '''def : op
           | fun'''
    p[0] = [p[1]]

def p_op(p):
    'op : OP PYOP DCOLON sig ARROW sig'
    p[0] = opdef(p[2], sig(p[4],p[6]))

def p_fun(p):
    'fun : FUN NAME DCOLON sig ARROW sig'
    p[0] = fndef(p[2], sig(p[4],p[6]))


#------------------------------------------------------------------------

def p_sig1(p):
    "sig : '(' sig ')'"
    p[0] = p[2]

def p_sig2(p):
    "sig : '(' ')' "
    p[0] = ()

def p_sig3(p):
    'sig : NAME NAME'
    p[0] = pt(p[1], p[2])

def p_sig4(p):
    "sig : sig COMMA sig "
    p[0] = [p[1], p[3]]

def p_sig5(p):
    "sig : sig ARROW sig "
    p[0] = fn(p[1], p[3])

def p_sig6(p):
    'sig : NAME'
    p[0] = p[1]

#------------------------------------------------------------------------

def p_empty(t):
    'empty : '
    pass

#------------------------------------------------------------------------

def p_error(p):
    if p:
        raise ModuleSyntaxError(
            p.lineno,
            p.lexpos,
            '<stdin>',
            p.lexer.lexdata,
        )
    else:
        print("Syntax error at EOF")

#------------------------------------------------------------------------
# Module
#------------------------------------------------------------------------

#------------------------------------------------------------------------
# Toplevel
#------------------------------------------------------------------------

def debug_parse(data, lexer, parser):
    lexer.input(data)
    #while True:
        #tok = lexer.token()
        #if not tok: break
        #print tok
    return parser.parse(data)

def load_parser(debug=False):
    if debug:
        from ply import lex, yacc
        path = os.path.relpath(__file__)
        dir_path = os.path.dirname(path)
        lexer = lex.lex(lextab="mlex", outputdir=dir_path, optimize=0)
        parser = yacc.yacc(tabmodule='myacc',outputdir=dir_path,
                write_tables=0, debug=0, optimize=0)
        return partial(debug_parse, lexer=lexer, parser=parser)
    else:
        module = sys.modules[__name__]
        lexer = lexfrom(module, mlex)
        parser = yaccfrom(module, myacc, lexer)

        # curry the lexer into the parser
        return partial(parser.parse, lexer=lexer)

def parse(pattern):
    parser = load_parser(debug=True)
    return parser(pattern)

def mopen(f):
    fd = open(f)
    contents = fd.read()
    b = parse(contents)

    return build_module(b)

def build_def(sig):
    return Definition(Signature(*sig))

def build_module(a):
    return Module([
        (iface.name , Interface(iface.name, iface.params, iface.body))
        for iface in a
    ])

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------

class Module(object):
    def __init__(self, insts):
        self.ifaces = dict(insts)

    def instantiate(self, name, base, params=None):
        return self.ifaces[name].instantiate(params)

    def __repr__(self):
        return pformat({
            'interfaces': self.ifaces
        })

class Instance(object):

    def __init__(self):
        self.symtab = {}

    def resolve(self, fn):
        ty = self.symtab[fn]
        return ty

class Interface(object):
    def __init__(self, name, params, defs):
        self.name = name
        self.params = params
        self.defs = dict([(name, build_def(sig)) for name, sig in defs])

    def instantiate(self, name, params):
        pass

    def __repr__(self):
        return pformat({
            'name'   : self.name,
            'params' : self.params,
            'defs'   : self.defs,
        })

class Definition(object):
    def __init__(self, sig):
        self.sig = sig

    def __repr__(self):
        return pformat({
            'sig': self.sig,
        })

class Signature(object):

    def __init__(self, dom, cod):
        self.dom = dom
        self.cod = cod

    def __repr__(self):
        return pformat({
            'dom': (self.dom),
            'cod': (self.cod),
        })

#------------------------------------------------------------------------

python = {
    'int'     : int,
    'float'   : float,
    'complex' : complex,
    'object'  : object,
}

if __name__ == '__main__':
    a = mopen('module/blaze.mod')
    print a

    # typeset

    # just has to inform the type of graph node that gets
    # outputted and the datashape of that node.

    # Entire thing needs to compile down into a giant dictionary
    # with the type mappings for a single concrete type.
    #a.instantiate('Ix', )

    import readline
    parser = load_parser(debug=True)
    readline.parse_and_bind('')

    while True:
        try:
            line = raw_input('>> ')
            print parser(line)
        except EOFError:
            break
