"""
The improved parser for Datashape grammar.

Grammar::

    statement ::= lhs_expression EQUALS rhs_expression
                | rhs_expression

    lhs_expression ::= lhs_expression SPACE lhs_expression
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
import re
from functools import partial
from collections import namedtuple

import ply.lex as lex
import ply.yacc as yacc

#------------------------------------------------------------------------
# Errors
#------------------------------------------------------------------------

syntax_error = """

  File {filename}, line {lineno}
    {line}
    {pointer}

DatashapeSyntaxError: {msg}
"""

class DatashapeSyntaxError(Exception):
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
        return syntax_error.format(**{
            'filename' : self.filename,
            'lineno'   : self.lineno,
            'line'     : self.text,
            'pointer'  : ' '*self.col_offset + '^',
            'msg'      : self.msg
        })

#------------------------------------------------------------------------
# Lexer
#------------------------------------------------------------------------

tokens = (
    'SPACE','NAME', 'NUMBER', 'EQUALS', 'COMMA', 'COLON',
    'LBRACE', 'RBRACE'
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

t_NAME   = r'[a-zA-Z_][a-zA-Z0-9_]*'
t_EQUALS = r'='
t_COMMA  = r','
t_COLON  = r':'
t_LBRACE = r'\{'
t_RBRACE = r'\}'
t_ignore = '\n'

def t_SPACE(t):
    r'\s'
    return t

def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_error(t):
    print("Unknown token '%s'" % t.value[0])
    t.lexer.skip(1)

# Build the lexer
#lexer = lex.lex()

#------------------------------------------------------------------------
# Parser
#------------------------------------------------------------------------

precedence = (
    ('right' , 'COMMA'),
    ('left'  , 'SPACE'),
)

tyinst     = namedtuple('tyinst', 'conargs')
tydecl     = namedtuple('tydecl', 'lhs, rhs')
simpletype = namedtuple('simpletype', 'nargs, tycon, tyvars')

def p_statement_assign(p):
    'statement : lhs_expression EQUALS rhs_expression'
    constructid = p[1][0]
    parameters  = p[1][1:]
    rhs         = p[3]

    lhs = simpletype(len(parameters), constructid, parameters)
    p[0] = tydecl(lhs, rhs)

def p_statement_expr(p):
    'statement : rhs_expression'
    p[0] = tyinst(p[1])

def p_lhs_expression(p):
    'lhs_expression : lhs_expression SPACE lhs_expression'
    # tuple addition
    p[0] = p[1] + p[3]

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
    'record_opt : record_opt COMMA record_opt'
    p[0] = [p[1], p[3]]

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
# Whitespace Preprocessor
#------------------------------------------------------------------------

def compose(f, g):
    return lambda x: g(f(x))

# remove trailing and leading whitespace
pass1 = re.compile(
    '^\s*'
    '|\s*$'
)
pre_trailing = partial(pass1.sub, '')

# collapse redundent whitespace
pass2 = re.compile(r'\s+')
pre_whitespace = partial(pass2.sub, ' ')

# push to normal form for whitespace around the equal sign
pass3 = re.compile(
    ' = ' # a b = c d -> a b=c d
    '| =' # a b =c d  -> a b=c d
    '|= ' # a b= c d  -> a b=c d
)
equal_trailing = partial(pass3.sub,'=')

# Strip whitespace on the right side
def rhs_strip(s):
    if s.find('=') > -1:
        try:
            lhs, rhs = s.split('=')
            return '='.join([lhs, rhs.replace(' ','')])
        except ValueError:
            # it's probably malformed, let the parser catch it
            return s
    else:
        return s.replace(' ','')

preparse = reduce(compose, [
    pre_whitespace,
    pre_trailing,
    equal_trailing,
    rhs_strip
])

#------------------------------------------------------------------------
# Toplevel
#------------------------------------------------------------------------

# TODO: deprecated??
# def datashape_parser(s, debug=False):
#     inputs = preparse(s)
#     ast = parser.parse(inputs,lexer=lexer)
#     return ast

def datashape_pprint(ast, depth=0):
    raise NotImplementedError

def make_parser():
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)

    lexer = lex.lex(lextab="dshape_lexer")
    parser = yacc.yacc(tabmodule='dshape_yacc',outputdir=dir_path,debug=0,
        write_tables=0)

    return parser
