import re
from ply.lex import lex

from .errors import error

tokens = [
 'ARROW', 'ASSIGN', 'COLON', 'COMMA', 'CONST', 'DEF', 'DIVIDE', 'ELSE', 'EQ',
 'FALSE', 'FLOAT', 'FOR', 'FOREIGN', 'GE', 'GT', 'ID', 'IF', 'IN', 'INTEGER',
 'LAND', 'LBRACE', 'LBRACKET', 'LE', 'LOR', 'LPAREN', 'LT', 'MINUS',
 'NE', 'NOT', 'PLUS', 'PRINT', 'RANGE', 'RBRACE', 'RBRACKET', 'RETURN',
 'RPAREN', 'SEMI', 'SHL', 'SHR', 'STRING', 'TIMES', 'TRUE', 'VAR', 'WHILE',
 'DOT',
]

reserved = set([
    'False',
    'True',
    # --
    'const',
    'def',
    'else',
    'for',
    'foreign',
    'if',
    'in',
    'print',
    'range',
    'return',
    'var',
    'while'
])

_escapes_re = r'(\\b[0-9a-fA-F]{2})|(\\.)'
_escape_map = {
    r'\n' : '\n',  # newline
    r'\t' : '\t',  # tab
    r'\r' : '\r',  # carriage return
    r'\\' : '\\',  # backslash
    r'\"' : '"',   # quote
}
_escape_pat = re.compile(_escapes_re)

t_ignore = ' \t\r'

t_FALSE     = r'False'
t_TRUE      = r'True'

t_PLUS      = r'\+'
t_MINUS     = r'-'
t_TIMES     = r'\*'
t_DIVIDE    = r'/'
t_ASSIGN    = r'='
t_SEMI      = r'\;'
t_LPAREN    = r'\('
t_RPAREN    = r'\)'
t_COMMA     = r','
t_COLON     = r':'
t_ARROW     = r'->'

t_LT        = r'<'
t_GT        = r'>'
t_LE        = r'<='
t_GE        = r'>='
t_EQ        = r'=='
t_NE        = r'!='
t_DOT       = r'\.'

t_SHL       = '<<'
t_SHR       = '>>'

t_LAND      = r'&&'
t_LOR       = r'\|\|'
t_NOT       = r'!'

t_LBRACKET  = r'\['
t_RBRACKET  = r'\]'
t_LBRACE    = r'\{'
t_RBRACE    = r'\}'

def t_FLOAT(t):
    r'(([0-9]+(\.[0-9]*)?[eE][\+-]?[0-9]+)|(\.[0-9]+([eE][\+-]?[0-9]+)?)|([0-9]+\.[0-9]*))'
    t.value = float(t.value)
    return t

def t_INTEGER(t):
    r'(0|0x|0X)?\d+'
    if t.value.startswith(('0x','0X')):
        t.value = int(t.value,16)
    elif t.value.startswith('0'):
        t.value = int(t.value,8)
    else:
        t.value = int(t.value)
    return t

def t_STRING(t):
    r'\"((\\.)|[^\\\n])*?\"'
    t.value = t.value[1:-1]
    _escape_token(t)
    return t

def t_ID(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    if t.value in reserved:
        t.type = t.value.upper()
    return t

#------------------------------------------------------------------------

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_COMMENT(t):
    r'\#.*'
    t.lexer.lineno += t.value.count('\n')

def t_error(t):
    error(t.lexer.lineno,"Illegal character %r" % t.value[0])
    t.lexer.skip(1)

#------------------------------------------------------------------------
# String Escaping
#------------------------------------------------------------------------

class Unescaped(Exception): pass

def escape_token(m):
    escape_code = m.group()
    if escape_code[0:2] == '\\b' and len(escape_code) == 4:
        return chr(int(escape_code[2:],16))
    if escape_code in _escape_map:
        return _escape_map[escape_code]
    else:
        raise Unescaped(escape_code)

def _escape_token(t):
    try:
        t.value = _escape_pat.sub(escape_token, t.value)
    except Unescaped as e:
        escape_code = e.args[0]
        error(t.lexer.lineno,"Syntax Error: Unescaped sequence '%s'" % escape_code)
        return escape_code
