"""
The improved parser for Datashape grammar.

Grammar::

    top : mod
        | stmt

    mod : mod mod
        | stmt

    stmt : TYPE lhs_expression EQUALS rhs_expression
         | rhs_expression

    lhs_expression : lhs_expression lhs_expression
                   | NAME

    rhs_expression : rhs_expression COMMA rhs_expression
                   | appl
                   | record
                   | BIT
                   | NAME
                   | NUMBER

    appl : NAME '(' rhs_expression ')'

    record : LBRACE record_opt RBRACE
    record_opt : record_opt SEMI record_opt
    record_opt : record_item
    record_opt : empty
    record_item : NAME COLON '(' rhs_expression ')'
    record_item : NAME COLON BIT
                : NAME COLON NAME
                | NAME COLON NUMBER
    empty :

"""

import os
import sys

from functools import partial
from collections import namedtuple

try:
    import dlex
    import dyacc
    DEBUG = False
except:
    DEBUG = True

from blaze.plyhacks import yaccfrom, lexfrom
from blaze.error import CustomSyntaxError

#------------------------------------------------------------------------
# Errors
#------------------------------------------------------------------------

class DatashapeSyntaxError(CustomSyntaxError):
    pass

#------------------------------------------------------------------------
# Lexer
#------------------------------------------------------------------------

tokens = (
    'TYPE', 'NAME', 'NUMBER', 'EQUALS', 'COMMA', 'COLON',
    'LBRACE', 'RBRACE', 'SEMI', 'BIT'
)

literals = [
    '=' ,
    ',' ,
    '(' ,
    ')' ,
    ':' ,
    '{' ,
    '}' ,
]

bits = set([
    'bool',
    'int8',
    'int16',
    'int32',
    'int64',
    'int64',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
    'uint64',
    'uint64',
    'float16',
    'float32',
    'float64',
    'float128',
    'complex64',
    'complex128',
    'complex256',
    'object',
    'datetime64',
    'timedelta64',
])

t_EQUALS = r'='
t_COMMA  = r','
t_COLON  = r':'
t_SEMI   = r';'
t_LBRACE = r'\{'
t_RBRACE = r'\}'
t_ignore = '[ ]'

def t_TYPE(t):
    r'type'
    return t

def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")

def t_NAME(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    if t.value in bits:
        t.type = 'BIT'
    return t

def t_COMMENT(t):
    r'\#.*'
    pass

def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_error(t):
    print("Unknown token '%s'" % t.value[0])
    t.lexer.skip(1)

#------------------------------------------------------------------------
# Parser
#------------------------------------------------------------------------

precedence = (
    ('right' , 'COMMA'),
)

bittype     = namedtuple('bit', 'name')
tyinst     = namedtuple('tyinst', 'conargs')
tydecl     = namedtuple('tydecl', 'lhs, rhs')
tyappl     = namedtuple('tyappl', 'head, args')
simpletype = namedtuple('simpletype', 'nargs, tycon, tyvars')

def p_top(p):
    '''top : mod
           | stmt
    '''
    p[0] = p[1]

#------------------------------------------------------------------------

def p_decl1(p):
    'mod : mod mod'
    p[0] = [p[1], p[2]]

def p_decl2(p):
    'mod : stmt'
    p[0] = p[1]

#------------------------------------------------------------------------

def p_statement_assign(p):
    'stmt : TYPE lhs_expression EQUALS rhs_expression'

    # alias
    if len(p[2]) == 1:
        constructid = p[2][0]
        parameters  = ()
        rhs         = p[4]

    # paramaterized
    else:
        constructid = p[2][0]
        parameters  = p[2][1:]
        rhs         = p[4]

    lhs = simpletype(len(parameters), constructid, parameters)
    p[0] = tydecl(lhs, rhs)

def p_statement_expr(p):
    'stmt : rhs_expression'
    p[0] = tyinst(p[1])

#------------------------------------------------------------------------

def p_lhs_expression(p):
    'lhs_expression : lhs_expression lhs_expression'
    # tuple addition
    p[0] = p[1] + p[2]

def p_lhs_expression_node(p):
    'lhs_expression : NAME'
    p[0] = (p[1],)

#------------------------------------------------------------------------

def p_rhs_expression_node1(p):
    '''rhs_expression : appl
                      | record'''
    p[0] = p[1]

def p_rhs_expression_node2(p):
    '''rhs_expression : BIT'''
    p[0] = (bittype(p[1]),)

def p_rhs_expression_node3(p):
    '''rhs_expression : NAME
                      | NUMBER'''
    p[0] = (p[1],)

def p_rhs_expression(p):
    'rhs_expression : rhs_expression COMMA rhs_expression'''
    # tuple addition
    p[0] = p[1] + p[3]

#------------------------------------------------------------------------

def p_appl(p):
    "appl : NAME '(' rhs_expression ')'"
    p[0] = (tyappl(p[1], p[3]),)

def p_record(p):
    'record : LBRACE record_opt RBRACE'

    if isinstance(p[2], list):
        p[0] = p[2]
    else:
        p[0] = (p[2],)

def p_record_opt1(p):
    'record_opt : record_opt SEMI record_opt'
    if p[3]: # trailing semicolon
        p[0] = [p[1], p[3]]
    else:
        p[0] = p[1]

def p_record_opt2(p):
    'record_opt : record_item'
    p[0] = p[1]

def p_record_opt3(p):
    'record_opt : empty'
    pass

def p_record_item1(p):
    "record_item : NAME COLON '(' rhs_expression ')' "
    p[0] = (p[1], p[4])

def p_record_item2(p):
    '''record_item : NAME COLON BIT
                   | NAME COLON NAME
                   | NAME COLON NUMBER
                   | NAME COLON record'''
    p[0] = (p[1], p[3])

#------------------------------------------------------------------------

def p_empty(t):
    'empty : '
    pass

def p_error(p):
    if p:
        raise DatashapeSyntaxError(
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

class Module(object):

    def __init__(self, **kwargs):
        # TODO: EVIL! just for debugging
        self.__dict__.update(kwargs)

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

#------------------------------------------------------------------------
# Toplevel
#------------------------------------------------------------------------

def build_ds(ds):
    return ds

def debug_parse(data, lexer, parser):
    lexer.input(data)
    while True:
        tok = lexer.token()
        if not tok: break
        print tok
    return parser.parse(data)

def load_parser(debug=False):
    if debug:
        from ply import lex, yacc
        path = os.path.relpath(__file__)
        dir_path = os.path.dirname(path)
        lexer = lex.lex(lextab="dlex", outputdir=dir_path, optimize=1)
        parser = yacc.yacc(tabmodule='dyacc',outputdir=dir_path,
                write_tables=1, debug=0, optimize=1)
        return partial(debug_parse, lexer=lexer, parser=parser)
    else:
        module = sys.modules[__name__]
        lexer = lexfrom(module, dlex)
        parser = yaccfrom(module, dyacc, lexer)

        # curry the lexer into the parser
        return partial(parser.parse, lexer=lexer)

def parse(pattern):
    parser = load_parser()
    res = parser(pattern)

    ds = build_ds(res)
    return ds

if __name__ == '__main__':
    import readline
    parser = load_parser()
    readline.parse_and_bind('')

    while True:
        try:
            line = raw_input('>> ')
            print parser(line)
        except EOFError:
            break
