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
    'SPACE','NAME','EQUALS', 'COMMA'
)

literals = [
    '=' ,
    ',' ,
    '(' ,
    ')' ,
    ':' ,
]

t_NAME   = r'[a-zA-Z_][a-zA-Z0-9_]*'
t_EQUALS = r'='
t_COMMA  = r','

def t_NEWLINE(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")

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
lexer = lex.lex()

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

def p_rhs_expression(p):
    'rhs_expression : rhs_expression COMMA rhs_expression'''
    # tuple addition
    p[0] = p[1] + p[3]

def p_lhs_expression_node(p):
    "lhs_expression : NAME"
    p[0] = (p[1],)

def p_rhs_expression_node(p):
    "rhs_expression : NAME"
    p[0] = (p[1],)

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

# Build the parser
parser = yacc.yacc()

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

def datashape_pprint(ast, level=0):
    raise NotImplementedError

def datashape_parser(s):
    inputs = preparse(s)
    ast = parser.parse(inputs,lexer=lexer)
    return ast

if __name__ == '__main__':
    print datashape_parser('a')
    print datashape_parser('a, b')
    print datashape_parser('a, b, c')
    print datashape_parser('a,      b')
    print datashape_parser('a,      b  ,     d')
    print datashape_parser('foo = c')
    print datashape_parser('a b c = d,e,f')
    print datashape_parser('   a b c = d, e   ')
    print datashape_parser('foo a b = c, d')
    print datashape_parser('foo a b = c,d,e')
    print datashape_parser('foo a b = c,d,e,   f')
    print datashape_parser('foo a b = c,   d,   e,   f')
    print datashape_parser('foo b = c,   d,   e,   f')
    print datashape_parser('a b c = d, e')
