import os
import re
from collections import namedtuple

import ply.lex as lex
import ply.yacc as yacc

parser = None

#------------------------------------------------------------------------
# Errors
#------------------------------------------------------------------------

syntax_error = """

  File {filename}, line {lineno}
    {line}
    {pointer}

ATermSyntaxError: {msg}
"""

class AtermSyntaxError(SyntaxError):
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
# AST Nodes
#------------------------------------------------------------------------

aterm = namedtuple('aterm', ('term', 'annotation'))
astr  = namedtuple('astr', ('val',))
aint  = namedtuple('aint', ('val',))
areal = namedtuple('areal', ('val',))
aappl = namedtuple('aappl', ('spine', 'args'))
atupl = namedtuple('atupl', ('args'))
aplaceholder = namedtuple('aplaceholder', ('type','args'))

INT  = 0
STR  = 1
BLOB = 2
TERM = 3
APPL = 4
LIST = 5

placeholders = {
    'appl': aappl,
    'str': astr,
    'int': aint,
    'real': areal,
    'term': (aterm, aappl, astr, aint, areal),
    'placeholder': aplaceholder,
    #'list' : alist
}

unquote = re.compile('"(?:[^\']*)\'|"([^"]*)"')

#------------------------------------------------------------------------
# Lexer
#------------------------------------------------------------------------

tokens = (
    'NAME', 'INT', 'DOUBLE', 'QUOTE', 'PLACEHOLDER', 'STRING'
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
t_QUOTE  = r'"'
t_ignore = '\x20\x09\x0A\x0D'

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
    p[0] = [p[1]]

def p_appl_value2(p):
    "appl_value : appl_value ',' appl_value"
    p[0] = p[1] + p[3]

#--------------------------------

def p_list(p):
    "list : '[' list_value ']' "
    p[0] = p[2]

def p_list_value1(p):
    "list_value : expr"
    p[0] = [p[1]]

def p_list_value2(p):
    "list_value : list_value ',' list_value"
    p[0] = p[1] + p[3]

#--------------------------------

def p_tuple(p):
    "tuple : '(' list_value ')' "
    p[0] = atupl(p[2])

def p_tuple_value1(p):
    "tuple_value : expr"
    p[0] = [p[1]]

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


#--------------------------------

def init(xs):
    for x in xs:
        return x

def tail(xs):
    for x in reversed(xs):
        return x

def aterm_iter(term):
    if isinstance(term, (aint, areal, astr)):
        yield term.val

    elif isinstance(term, aappl):
        yield term.spine
        for arg in term.args:
            for t in aterm_iter(arg):
                yield arg

    elif isinstance(term, aterm):
        yield term.term

    else:
        raise NotImplementedError

def aterm_zip(a, b):
    if isinstance(a, (aint, areal, astr)) and isinstance(b, (aint, areal, astr)):
        yield a.val == b.val, None

    elif isinstance(a, aappl) and isinstance(b, aappl):
        if len(a.args) == len(b.args):
            yield a.spine == b.spine, None
            for ai, bi in zip(a.args, b.args):
                for aj in aterm_zip(ai,bi):
                    yield aj
        else:
            yield False, None

    elif isinstance(a, aterm) and isinstance(b, aterm):
        yield a.term == b.term, None
        yield a.annotation == b.annotation, None

    elif isinstance(a, aplaceholder):
        # <appl(...)>
        if a.args:
            if isinstance(b, aappl):
                yield True, b.spine
                for ai, bi in zip(a.args, b.args):
                    for a in aterm_zip(ai,bi):
                        yield a
            else:
                yield False, None
        # <term>
        else:
            yield isinstance(b, placeholders[a.type]), b
    else:
        yield False, None


# left-to-right substitution
def aterm_splice(a, elts):

    if isinstance(a, (aint, areal, astr)):
        yield a

    elif isinstance(a, aappl):
        yield aappl(a.spine, [init(aterm_splice(ai,elts)) for ai in a.args])

    elif isinstance(a, aterm):
        yield a

    elif isinstance(a, aplaceholder):
        # <appl(...)>
        if a.args:
            spine = elts.pop()
            yield aappl(spine, [init(aterm_splice(ai,elts)) for ai in a.args])
        # <term>
        else:
            yield elts.pop()
    else:
        raise NotImplementedError

#--------------------------------

def has_prop(term, prop):
    return prop in term.annotation

#--------------------------------

def _init():
    global parser
    if not parser:
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)

        lexer = lex.lex(lextab="alex")
        parser = yacc.yacc(tabmodule='ayacc',outputdir=dir_path,debug=0,
            write_tables=0)
    else:
        parser = parser
    return parser

def parse(pattern):
    parser = _init()
    return parser.parse(pattern)

def match(pattern, subject, *captures):
    parser = _init()

    captures = []

    p = parser.parse(pattern)
    s = parser.parse(subject)

    for matches, capture in aterm_zip(p,s):
        if not matches:
            return False, []
        elif matches and capture:
            captures += [capture]
    return True, captures

def matches(pattern, subject):
    #parser = _init()

    p = pattern
    s = subject

    for matches, capture in aterm_zip(p,s):
        if not matches:
            return False
    return True

def build(pattern, *values):
    parser = _init()

    p = parser.parse(pattern)
    return list(aterm_splice(p,list(values)))
