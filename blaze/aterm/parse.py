"""
ATerm parser

t : bt                 -- basic term
  | bt {ty,m1,...}     -- annotated term

bt : C                 -- constant
   | C(t1,...,tn)      -- n-ary constructor
   | (t1,...,tn)       -- n-ary tuple
   | [t1,...,tn]       -- list
   | "ccc"             -- quoted string ( explicit double quotes )
   | int               -- integer
   | real              -- floating point number
"""

import re
import os

import ply.lex as lex
import ply.yacc as yacc

from terms import *

DEBUG = True

#------------------------------------------------------------------------
# Errors
#------------------------------------------------------------------------

parser = None

syntax_error = """

  File {filename}, line {lineno}
    {line}
    {pointer}

ATermSyntaxError: {msg}
"""

class AtermSyntaxError(Exception):
    """
    Makes aterm parse errors look like Python SyntaxError.
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
            msg      = self.msg
        )

#------------------------------------------------------------------------
# Lexer
#------------------------------------------------------------------------

tokens = (
    'NAME', 'INT', 'DOUBLE', 'PLACEHOLDER', 'STRING'
)

literals = [
    ',' ,
    '(' ,
    ')' ,
    '{' ,
    '}' ,
    '<' ,
    '>' ,
    '[' ,
    ']' ,
]

t_NAME   = r'[a-zA-Z_][a-zA-Z0-9_]*'
t_ignore = '\x20\x09\x0A\x0D'

unquote = re.compile('"(?:[^\']*)\'|"([^"]*)"')

def t_DOUBLE(t):
    r'\d+\.(\d+)?'
    t.value = float(t.value)
    return t

def t_PLACEHOLDER(t):
    r'appl|str|int|real|placeholder|appl|list|term'
    return t

def t_INT(t):
    r'\d+'
    try:
        t.value = int(t.value)
    except ValueError:
        print("Integer value too large %d", t.value)
        t.value = 0
    return t

def t_STRING(t):
    r'"([^"\\]|\\.)*"'
    t.value = t.value.encode('ascii')
    t.value = unquote.findall(t.value)[0]
    return t

def t_error(t):
    print("Unknown token '%s'" % t.value[0])
    t.lexer.skip(1)

# Top of the parser
def p_expr1(p):
    """expr : avalue
            | value"""
    p[0] = p[1]

#--------------------------------

def p_avalue(p):
    "avalue : value '{' annotation '}'"
    p[0] = aterm(p[1], p[3])

def p_value(p):
    """value : term
             | appl
             | list
             | tuple
             | string
             | placeholder
             | empty"""
    p[0] = p[1]

#--------------------------------

def p_term_double(p):
    "term : DOUBLE"
    p[0] = areal(p[1])

def p_term_int(p):
    "term : INT"
    p[0] = aint(p[1])

def p_term_term(p):
    "term : NAME"
    p[0] = aterm(p[1], None)

#--------------------------------

# Terms in annotations must not be themselves be annotated, i.e.
# they must be ``value``, not ``avalue``.
def p_annotation1(p):
    "annotation : value"
    p[0] = (p[1],)

def p_annotation2(p):
    "annotation : annotation ',' annotation"
    p[0] = (p[1], p[3])

#--------------------------------

def p_appl(p):
    "appl : term '(' appl_value ')' "
    p[0] = aappl(p[1], p[3])

def p_appl_value1(p):
    "appl_value : expr"
    if p[1]:
        p[0] = [p[1]]
    else:
        p[0] = []

def p_appl_value2(p):
    "appl_value : appl_value ',' appl_value"
    p[0] = p[1] + p[3]

#--------------------------------

def p_list(p):
    "list : '[' list_value ']' "
    p[0] = alist(p[2])

def p_list_value1(p):
    "list_value : expr"
    if p[1]:
        p[0] = [p[1]]
    else:
        p[0] = []

def p_list_value2(p):
    "list_value : list_value ',' list_value"
    p[0] = p[1] + p[3]

#--------------------------------

def p_tuple(p):
    "tuple : '(' tuple_value ')' "
    p[0] = atupl(p[2])

def p_tuple_value1(p):
    "tuple_value : expr"
    if p[1]:
        p[0] = [p[1]]
    else:
        p[0] = []

def p_tuple_value2(p):
    "tuple_value : tuple_value ',' tuple_value"
    p[0] = p[1] + p[3]

#--------------------------------

def p_string(p):
    "string : STRING"
    p[0] = astr(p[1])

#--------------------------------

def p_placeholder1(p):
    "placeholder : '<' PLACEHOLDER '(' appl_value ')' '>'"
    p[0] = aplaceholder(p[2], p[4])

def p_placeholder2(p):
    "placeholder : '<' PLACEHOLDER  '>'"
    p[0] = aplaceholder(p[2], None)

#--------------------------------

def p_empty(t):
    "empty : "
    pass

#--------------------------------

def p_error(p):
    if p:
        raise AtermSyntaxError(
            p.lineno,
            p.lexpos,
            '<stdin>',
            p.lexer.lexdata,
        )
    else:
        raise SyntaxError("Syntax error at EOF")

#------------------------------------------------------------------------
# Toplevel
#------------------------------------------------------------------------

def make_parser():
    global parser
    if not parser:
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)

        lexer = lex.lex(lextab="alex")
        parser = yacc.yacc(tabmodule='ayacc',outputdir=dir_path)
    else:
        parser = parser
    return parser

def parse(pattern):
    parser = make_parser()
    return parser.parse(pattern)
