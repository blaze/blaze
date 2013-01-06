"""
The improved parser for Datashape grammar.

Grammar::

    statement ::= TYPE lhs_expression EQUALS rhs_expression
                | rhs_expression

    lhs_expression ::= lhs_expression lhs_expression
                     | NAME

    rhs_expression ::= rhs_expression COMMA rhs_expression

    rhs_expression ::= record
                     | NAME
                     | NUMBER

    record ::= LBRACE record_opt RBRACE

    record_opt ::= record_opt COMMA record_opt
                 | record_item
                 | empty

    record_item ::= NAME COLON '(' rhs_expression ')'
                  | NAME COLON NAME
                  | NAME COLON NUMBER

    empty ::=

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
    'LBRACE', 'RBRACE', 'SEMI'
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

tyinst     = namedtuple('tyinst', 'conargs')
tydecl     = namedtuple('tydecl', 'lhs, rhs')
simpletype = namedtuple('simpletype', 'nargs, tycon, tyvars')

def p_decl1(p):
    'decl : decl decl'
    p[0] = [p[1], p[2]]

def p_decl2(p):
    'decl : statement'
    p[0] = p[1]

def p_statement_assign(p):
    'statement : TYPE lhs_expression EQUALS rhs_expression'
    if len(p[3]) == 1:
        constructid = p[2][0]
        parameters  = ()
        rhs         = p[4]
    else:
        constructid = p[2][0]
        parameters  = p[2][1:]
        rhs         = p[4]

    lhs = simpletype(len(parameters), constructid, parameters)
    p[0] = tydecl(lhs, rhs)

def p_statement_expr(p):
    'statement : rhs_expression'
    p[0] = tyinst(p[1])

def p_lhs_expression(p):
    'lhs_expression : lhs_expression lhs_expression'
    # tuple addition
    p[0] = p[1] + p[2]

def p_lhs_expression_node(p):
    "lhs_expression : NAME"
    p[0] = (p[1],)

def p_rhs_expression(p):
    'rhs_expression : rhs_expression COMMA rhs_expression'''
    # tuple addition
    p[0] = p[1] + p[3]

def p_rhs_expression_node1(p):
    '''rhs_expression : record'''
    p[0] = p[1]

def p_rhs_expression_node2(p):
    '''rhs_expression : NAME
                      | NUMBER'''
    p[0] = (p[1],)

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
    '''record_item : NAME COLON NAME
                   | NAME COLON NUMBER'''
    p[0] = (p[1], p[3])

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

class Module:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

#------------------------------------------------------------------------
# Toplevel
#------------------------------------------------------------------------


def debug_parse(data, lexer, parser):
    lexer.input(data)
    while True:
        tok = lexer.token()
        if not tok: break
        print tok
    print parser.parse(data)

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
    return parser(pattern)

if __name__ == '__main__':
    import readline
    parser = load_parser(debug=True)
    readline.parse_and_bind('')

    while True:
        try:
            line = raw_input('>> ')
            print parse(line)
        except EOFError:
            break
