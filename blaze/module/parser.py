# -*- coding: utf-8 -*-

import re
import os
import sys
import logging

from functools import partial

import nodes as N
from typing import build_module, opts

try:
    import mlex
    import myacc
    DEBUG = False
except:
    DEBUG = True

from plyhacks import yaccfrom, lexfrom

#------------------------------------------------------------------------
# Errors
#------------------------------------------------------------------------

syntax_error = """
  File {filename}, line {lineno}
    {line}

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
            msg      = self.msg,
            error    = self.__class__.__name__,
        )

#------------------------------------------------------------------------
# Lexer
#------------------------------------------------------------------------

tokens = (
    'INTERFACE', 'INT', 'NAME', 'ARROW', 'COLON', 'OP', 'FUN', 'DCOLON',
    'PYOP', 'COMMA', 'EFFECT', 'SORTS', 'MODULE', 'TILDE',
    'USE', 'QUALIFIED', 'EL', 'TY', 'FOREIGN', 'IN', 'TYPESET',
    'QUOTE', 'CINCLUDE', 'FOR'
)

effects = [
    'alloc',
    'dealloc,'
    'read',
    'assign,'
    'dirty',
]

literals = [
    '=' , ',' , '(' , ')' , '{' , '}' , '!' , '[' , ']' , '|' ,
    '$' , '*' ,
]

reserved = {
    'in' : 'IN',
    'op' : 'OP',
    'el' : 'EL',
    'ty' : 'TY',
}

# comma and arrow operators are right associative
# and have kind ( * -> * -> * )

precedence = (
    ('right', 'ARROW'),
    ('right', 'COMMA'),
)

pyops = [
  '+' , '*' , '-' , '/' , '**' , '==' , '!=' , '//' , '<' , '>' ,
  '%' , '&' , '|' , '^' , '>=' , '<=' , '<<' , '>>' ,
]

t_COMMA = r','
t_ignore = ' \t\r'

def t_MODULE(t):
    r'module'
    return t

def t_INTERFACE(t):
    r'trait|impl'
    return t

def t_PYOP(t):
    return t

t_PYOP.__doc__ = r'|'.join(re.escape('_%s_') % s for s in pyops)

def t_OP(t):
    r'op'
    return t

def t_FUN(t):
    r'fun'
    return t

def t_FOR(t):
    r'for'
    return t

def t_EL(t):
    r'el'
    return t

def t_FOREIGN(t):
    r'foreign'
    return t

def t_TYPESET(t):
   r'typeset'
   return t

def t_TY(t):
    r'ty'
    return t

def t_USE(t):
    r'use'
    return t

def t_CINCLUDE(t):
    r'include'
    return t

def t_QUALIFIED(t):
    r'qualified'
    return t

def t_SORTS(t):
    r'sorts'
    return t

def t_DCOLON(t):
    r'::'
    return t

def t_QUOTE(t):
    r'\"'
    return t

def t_TILDE(t):
    r'\~'
    return t

def t_COLON(t):
    r'\:'
    return t

def t_ARROW(t):
    r'->'
    return t

def t_EFFECT(t):
    r'alloc|dealloc|read|assign|dirty'
    return t

# unqualified names : A
# qualified names   : A.b.c

def t_INT(t):
    r'[0-9]+'
    return t

def t_NAME(t):
    r'[a-zA-Z_][a-zA-Z0-9_\.\']*'
    if t.value in reserved:
        t.type = reserved[t.value]
    return t

def t_COMMENT(t):
    r'\#.*'
    pass

def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")

def t_error(t):
    print("Unknown token '%s'" % t.value[0])
    t.lexer.skip(1)

#------------------------------------------------------------------------
# Parser
#------------------------------------------------------------------------

def p_top1(p):
    '''top : mod'''
    p[0] = [p[1]]

def p_top2(p):
    '''top : import top'''
    p[0] = [p[1]] + p[2]

def p_top3(p):
    '''top : empty'''
    pass

#------------------------------------------------------------------------

def p_mod(p):
    '''mod : MODULE NAME '{' mod '}' '''
    p[0] = N.mod(p[2], p[4])

def p_import(p):
    '''import : USE NAME
              | USE QUALIFIED NAME
              | CINCLUDE QUOTE NAME QUOTE'''
    if p[1] == 'USE':
        p[0] = N.imprt(False, p[2])
    elif p[1] == 'CINCLUDE':
        p[0] = N.include(p[3])

#------------------------------------------------------------------------

def p_decl1(p):
    'mod : mod mod'
    p[0] = p[1] + p[2]

def p_decl2(p):
    '''mod : stmt
           | sorts
           | typeset '''
    p[0] = [p[1]]

#------------------------------------------------------------------------

def p_statement_assign1(p):
    '''stmt : INTERFACE NAME iparams COLON def
            | INTERFACE NAME iparams COLON empty
    '''
    if p[1] == 'trait':
        defs = p[5] or []
        p[0] = N.tclass(p[2], p[3], defs)

    elif p[1] == 'impl':
        defs = p[5] or []
        meta = None
        p[0] = N.instance(p[2], p[3], defs, meta)

    else:
        raise NotImplementedError

def p_statement_assign2(p):
    '''stmt : INTERFACE NAME iparams FOR enum COLON def '''
    assert p[1] == 'impl'
    defs = p[7] or []
    meta = p[5]
    p[0] = N.instance(p[2], p[3], defs, meta)

#------------------------------------------------------------------------

def p_interface_params1(p):
    'iparams : "[" params "]" '
    p[0] = p[2]

def p_interface_params2(p):
    'iparams : "[" multiparams "]" '
    p[0] = p[2]

def p_enumeration_params(p):
    'enum : "(" tenum ")" '
    p[0] = p[2]

def p_typeset_enum1(p):
    "tenum : tenum COMMA tenum"
    p[0] = p[1] + p[3]

def p_typeset_enum2(p):
    "tenum : NAME IN NAME"
    p[0] = [N.tenum(p[1], p[3])]

#------------------------------------------------------------------------

def p_interface_params3(p):
    "iparams : empty "
    pass

def p_params1(p):
    "params : params params"
    p[0] = p[1] + p[2]

def p_params2(p):
    'params : NAME '
    p[0] = (p[1],)

def p_params3(p):
    'multiparams : params COMMA params'
    p[0] = (p[1],p[3])

def p_def1(p):
    'def : def def'
    p[0] = p[1] + p[2]

def p_def2(p):
    '''def : op
           | fun
           | sorts
           | el
           | ty'''
    p[0] = [p[1]]

#------------------------------------------------------------------------

# signature
def p_op1(p):
    'op : OP PYOP DCOLON sig ARROW sig'
    p[0] = N.opdef(p[2], N.sig(p[4], p[6]))

# operator alias
def p_op2(p):
    'op : OP PYOP TILDE NAME'
    p[0] = N.opalias(p[2], p[4])

# operator implementation
def p_op3(p):
    'op : OP PYOP "=" impl'
    p[0] = N.opimp(p[2], p[4])

#------------------------------------------------------------------------

# signature
def p_fun1(p):
    'fun : FUN NAME DCOLON sig ARROW sig'
    p[0] = N.fndef(p[2], N.sig(p[4],p[6]))

# definition
def p_fun2(p):
    'fun : FUN NAME "=" impl'
    p[0] = N.fnimp(p[2], p[4])

# calling convention
def p_calling(p):
    '''calling : NAME
               | empty'''
    p[0] = p[1]

# foreign definition
def p_fun3(p):
    'fun : FUN NAME "=" FOREIGN calling QUOTE NAME QUOTE sig ARROW sig'
    p[0] = N.foreign(p[2], p[5], p[6], N.fndef(p[9], p[11]))

#------------------------------------------------------------------------

def p_sorts1(p):
    'sorts : SORTS sorts'
    p[0] = N.sortdef(p[2])

def p_sorts2(p):
    'sorts : sorts COMMA sorts'
    p[0] = p[1] + p[3]

def p_sorts3(p):
    'sorts : NAME'
    p[0] = [p[1]]

#------------------------------------------------------------------------

def p_typeset1(p):
    'typeset : TYPESET NAME "=" typeset_list'
    p[0] = N.typeset(p[2], p[4])

def p_typeset2(p):
    'typeset_list : typeset_list "|" typeset_list '
    p[0] = p[1] + p[3]

def p_typeset3(p):
    'typeset_list : NAME'
    p[0] = [p[1]]

#------------------------------------------------------------------------

# constant
def p_el1(p):
    'el : EL NAME DCOLON sig'
    p[0] = N.eldef(p[2], p[4])

# foreign constant
def p_el2(p):
    'el : EL NAME "=" FOREIGN QUOTE NAME QUOTE sig'
    p[0] = N.eldef(p[2], p[7])

#------------------------------------------------------------------------

# class scope type
def p_ty(p):
    'ty : TY NAME DCOLON "*"'
    p[0] = N.tydef(p[2])

#------------------------------------------------------------------------

def p_sig1(p):
    "sig : '(' sig ')'"
    p[0] = p[2]

# void
def p_sig2(p):
    "sig : '(' ')' "
    p[0] = ()

def p_sig3(p):
    "sig : '!' EFFECT '(' ')' "
    p[0] = ()

def p_sig5(p):
    "sig : sig COMMA sig "
    p[0] = [p[1], p[3]]

def p_sig6(p):
    "sig : sig ARROW sig "
    p[0] = N.fn(p[1], p[3])

def p_sig7(p):
    '''sig : '!' EFFECT NAME'''
    p[0] = (p[1], p[2])

def p_sig8(p):
    'sig : NAME NAME NAME'
    p[0] = N.pt(p[1], [p[2], p[3]])

def p_sig9(p):
    'sig : NAME NAME'
    p[0] = N.pt(p[1], [p[2]])

def p_sig10(p):
    'sig : NAME'
    p[0] = p[1]

#------------------------------------------------------------------------

def p_impl(p):
    """impl : appl
            | term
    """
    p[0] = p[1]

def p_term_term1(p):
    "term : NAME"
    p[0] = p[1]

def p_term_term2(p):
    "term : '$' INT"
    p[0] = p[2]

def p_appl(p):
    "appl : term '(' val ')' "
    p[0] = (p[1], p[3])

#------------------------------------------------------------------------

def p_val1(p):
    "val : impl"
    if p[1]:
        p[0] = [p[1]]
    else:
        p[0] = []

def p_val2(p):
    "val : val COMMA val"
    p[0] = p[1] + p[3]

#------------------------------------------------------------------------

def p_empty(t):
    'empty : '
    pass

#------------------------------------------------------------------------

def p_error(p):
    line = p.lexer.lexdata.split('\n')[p.lineno-1]
    offset = 0
    column = p.lexpos - offset

    #print "Module".center(80, '=')
    #print p
    #print "".center(80, '=')

    if p:
        raise ModuleSyntaxError(
            p.lineno,
            column,
            '<stdin>',
            line,
        )
    else:
        raise SyntaxError("Syntax error at EOF")

#------------------------------------------------------------------------
# Module Construction
#------------------------------------------------------------------------

def debug_parse(data, lexer, parser):
    lexer.input(data)
    return parser.parse(data)

def load_parser(debug=False):
    if debug:
        from ply import lex, yacc
        path = os.path.abspath(__file__)
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

#------------------------------------------------------------------------
# Toplevel
#------------------------------------------------------------------------

def mread(s):
    """
    Read module from string
    """
    ast = parse(s)

    if opts.ddump_parse:
        logging.info(ast)

    return build_module(ast)

def mopen(f):
    """
    Read module from file
    """
    fd = open(f)
    contents = fd.read()
    ast = parse(contents)

    if opts.ddump_parse:
        logging.info(ast)

    return build_module(ast)

#------------------------------------------------------------------------

if __name__ == '__main__':
    a = mopen('debug.mod')
    print a.show()
