from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import ast

from collections import namedtuple
from . import coretypes as T

from ply import lex, yacc
from blaze.plyhacks import yaccfrom, lexfrom
from blaze.error import CustomSyntaxError

instanceof = lambda T: lambda X: isinstance(X, T)

#------------------------------------------------------------------------
# Errors
#------------------------------------------------------------------------

class DatashapeSyntaxError(CustomSyntaxError):
    pass

#------------------------------------------------------------------------
# Lexer
#------------------------------------------------------------------------

tokens = (
    'TYPE', 'NAME', 'NUMBER', 'STRING', 'STAR', 'EQUALS',
    'COMMA', 'COLON', 'LBRACE', 'RBRACE', 'SEMI', 'BIT',
    'VAR', 'JSON', 'DATA'
)

literals = [
    '=' ,
    '|' ,
    ',' ,
    '(' ,
    ')' ,
    ':' ,
    '{' ,
    '}' ,
    '*' ,
    '\!' ,
]

bits = set([
    'bool',
    'bytes',
    'int',
    'float',
    'int8',
    'int16',
    'int32',
    'int64',
    'int64',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
    'float16',
    'float32',
    'float64',
    'float128',
    'complex64',
    'complex128',
    'complex256',
    'string',
    'datetime64',
    'timedelta64',
])

t_EQUALS = r'='
t_COMMA  = r','
t_COLON  = r':'
t_SEMI   = r';'
t_LBRACE = r'\{'
t_RBRACE = r'\}'
t_STAR   = r'\*'
t_ignore = '[ ]'

def t_TYPE(t):
    r'type'
    return t

def t_DATA(t):
    r'data'
    return t

def t_newline(t):
    r'\n+'
    #t.lexer.lineno += t.value.count("\n")

def t_JSON(t):
    # Must be before NAME to match
    r'json'
    return t

def t_VAR(t):
    # Must be before NAME to match
    r'var'
    return t

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

def t_STRING(t):
    r'(?:"(?:[^"\n\r\\]|(?:\\x[0-9a-fA-F]{2})|(?:\\u[0-9a-fA-F]{4})|(?:\\.))*")|(?:\'(?:[^\'\n\r\\]|(?:\\x[0-9a-fA-F]+)|(?:\\u[0-9a-fA-F]{4})|(?:\\.))*\')'
    # Use the Python parser via the ast module to parse the string,
    # since neither the string_escape nor unicode_escape do the right thing
    if sys.version_info >= (3, 0):
        t.value = ast.parse(t.value).body[0].value.s
    else:
        t.value = ast.parse('u' + t.value).body[0].value.s
    return t

def t_error(t):
    raise Exception("Unknown token %s" % repr(t.value[0]))
    #t.lexer.skip(1)

#------------------------------------------------------------------------
# Parser
#------------------------------------------------------------------------

precedence = (
    ('right' , 'COMMA'),
)

dtdecl = namedtuple('dtdecl', 'name, elts')
tydecl = namedtuple('tydecl', 'lhs, rhs')
simpletype = namedtuple('simpletype', 'nargs, tycon, tyvars')

def p_top(p):
    '''top : mod '''
    p[0] = p[1]

#------------------------------------------------------------------------

def p_decl1(p):
    'mod : mod mod'
    p[0] = [p[1], p[2]]

def p_decl2(p):
    'mod : stmt'
    p[0] = p[1]

#------------------------------------------------------------------------

def p_data_assign(p):
    'stmt : DATA NAME EQUALS data'
    p[0] = dtdecl(p[2], p[4])

def p_statement_assign(p):
    'stmt : TYPE lhs_expression EQUALS rhs_expression'

    # alias
    if len(p[2]) == 1:
        constructid = p[2][0]
        parameters  = ()
        rhs         = p[4]

    # parameterized
    else:
        constructid = p[2][0]
        parameters  = p[2][1:]
        rhs         = p[4]

    lhs = simpletype(len(parameters), constructid, parameters)
    p[0] = tydecl(lhs, rhs)

def p_statement_expr(p):
    'stmt : rhs_expression'
    p[0] = p[1]

def p_enum_def1(p):
    "data : data '|' data"
    p[0] = p[1] + p[3]

def p_enum_def3(p):
    'data : NAME'
    p[0] = [p[1]]

#------------------------------------------------------------------------

def p_lhs_expression(p):
    'lhs_expression : lhs_expression lhs_expression'
    # tuple addition
    p[0] = p[1] + p[2]

def p_lhs_expression_node(p):
    'lhs_expression : NAME'
    p[0] = (p[1],)

#------------------------------------------------------------------------

def p_rhs_expression(p):
    'rhs_expression : rhs_expression_list'
    if len(p[1]) == 1:
        rhs = p[1][0]
        if getattr(rhs, 'cls', T.MEASURE) != T.MEASURE:
            raise TypeError('Only a measure can appear on the last position of a datashape, not %s' % repr(rhs))
        p[0] = rhs
    else:
        p[0] = T.DataShape(p[1])

def p_rhs_expression_list_node1(p):
    '''rhs_expression_list : appl
                           | record'''
    p[0] = (p[1],)

def p_rhs_expression_list__bit(p):
    '''rhs_expression_list : BIT'''
    p[0] = (T.Type._registry[p[1]],)

def p_rhs_expression_list__name(p):
    '''rhs_expression_list : NAME'''
    p[0] = (T.TypeVar(p[1]),)

def p_rhs_expression_list__number(p):
    '''rhs_expression_list : NUMBER'''
    p[0] = (T.Fixed(p[1]),)

def p_rhs_expression_list__var(p):
    '''rhs_expression_list : VAR'''
    p[0] = (T.Var(),)

def p_rhs_expression_list__json(p):
    '''rhs_expression_list : JSON'''
    p[0] = (T.JSON(),)

def p_rhs_expression_list__wild(p):
    '''rhs_expression_list : STAR'''
    p[0] = (T.Wild(),)

def p_rhs_expression_list(p):
    'rhs_expression_list : rhs_expression_list COMMA rhs_expression_list metadata'
    # tuple addition
    p[0] = p[1] + p[3]

#------------------------------------------------------------------------

def p_metadata1(p):
    "metadata : '!' LBRACE appl_args RBRACE "
    p[0] = p[3]

def p_metadata2(p):
    "metadata : empty"
    p[0] = None

#------------------------------------------------------------------------

def p_appl_args__appl__record(p):
    '''appl_args : appl
                 | record'''
    p[0] = (p[1],)

def p_appl_args__rhs_expression(p):
    "appl_args : '(' rhs_expression ')'"
    p[0] = (p[2],)

def p_appl_args__name(p):
    '''appl_args : NAME'''
    p[0] = (T.TypeVar(p[1]),)

def p_appl_args__bit(p):
    '''appl_args : BIT'''
    p[0] = (T.Type._registry[p[1]],)

def p_appl_args__number(p):
    '''appl_args : NUMBER'''
    p[0] = (T.IntegerConstant(p[1]),)

def p_appl_args__string(p):
    '''appl_args : STRING'''
    p[0] = (T.StringConstant(p[1]),)

def p_appl_args(p):
    'appl_args : appl_args COMMA appl_args'
    # tuple addition
    p[0] = p[1] + p[3]

#------------------------------------------------------------------------

def p_appl(p):
    """appl : NAME '(' appl_args ')'
            | BIT '(' appl_args ')'""" # BIT is here for 'string(...)'

    if p[1] == 'Categorical': # TODO: don't hardcode
        if not all(isinstance(x, T.TypeVar) for x in p[3]):
            raise Exception('Invalid categorical definition')

        p[0] = T.Enum(None, *p[3])
    if p[1] in reserved:
        # The appl_args part of the grammar already produces
        # TypeVar/IntegerConstant/StringConstant values
        p[0] = reserved[p[1]](*p[3])
    else:
        raise NameError('Cannot use the name %s for type application' % repr(p[1]))

#------------------------------------------------------------------------

def p_record(p):
    'record : LBRACE record_opt RBRACE'
    p[0] = T.Record(p[2])

def p_record_opt1(p):
    'record_opt : record_opt SEMI record_opt'
    p[0] = p[1] + p[3]

def p_record_opt2(p):
    'record_opt : record_item'
    p[0] = [p[1]]

def p_record_opt3(p):
    'record_opt : empty'
    p[0] = []

def p_record_name(p):
    '''record_name : NAME
                   | BIT
                   | TYPE
                   | JSON
                   | VAR'''
    p[0] = p[1]

def p_record_item1(p):
    "record_item : record_name COLON '(' rhs_expression ')'"
    p[0] = (p[1], p[4])

def p_record_item2(p):
    '''record_item : record_name COLON rhs_expression'''
    p[0] = (p[1], p[3])

#------------------------------------------------------------------------

def p_empty(t):
    'empty : '
    pass

def p_error(p):
    if p:
        raise DatashapeSyntaxError(
            p.lexpos,
            '<stdin>',
            p.lexer.lexdata,
        )
    else:
        raise DatashapeSyntaxError(
            0,
            '<stdin>',
            '',
        )

#------------------------------------------------------------------------
# Toplevel
#------------------------------------------------------------------------

reserved = {
    'Record'      : T.Record,
    'Range'       : T.Range,
    'Categorical' : T.Enum,
    'Option'      : T.Option,
    #'Either'   : T.Either,
    #'Union'    : T.Union,
    'string'   : T.String, # String type per proposal
    'Wild'     : T.Wild
}

def debug_parse(data, lexer, parser):
    lexer.input(data)
    while True:
        tok = lexer.token()
        if not tok: break
        print(tok)

    import logging
    logging.basicConfig(
        level = logging.DEBUG,
        filename = "parselog.txt",
        filemode = "w",
        format = "%(filename)10s:%(lineno)4d:%(message)s"
    )
    log = logging.getLogger()
    return parser.parse(data, debug=log)

#------------------------------------------------------------------------
# Toplevel
#------------------------------------------------------------------------

def rebuild():
    """ Rebuild the parser and lexer tables. """
    path = os.path.relpath(__file__)
    output = os.path.dirname(path)
    module = sys.modules[__name__]

    lex.lex(module=module, lextab="dlex", outputdir=output, debug=0, optimize=1)
    yacc.yacc(tabmodule='dyacc',outputdir=output, write_tables=1, debug=0, optimize=1)

    sys.stdout.write("Parse and lexer tables rebuilt.\n")

def _parse(source):
    try:
        from . import dlex
        from . import dyacc
    except ImportError:
        raise RuntimeError("Parse tables not built, run install script.")

    module = sys.modules[__name__]
    lexer  = lexfrom(module, dlex)
    parser = yaccfrom(module, dyacc, lexer)

    return parser.parse(source, lexer=lexer)

def parse_mod(pattern):
    dss = _parse(pattern)

    for ds in dss:
        if isinstance(ds, tydecl):
            yield ds.rhs
        elif isinstance(ds, dtdecl):
            yield T.Enum(ds.name, ds.elts)

def parse(pattern):
    ds = _parse(pattern)

    if isinstance(ds, dtdecl):
        raise TypeError('Predeclared categorical types are not allowed inline')

    # Just take the type from "type X = Y" statements
    if isinstance(ds, tydecl):
        if ds.lhs.nargs == 0:
            ds = ds.rhs
        else:
            raise TypeError('building a simple dshape with '
                            'type parameters is not supported')
    # Require that the type be concrete, not parameterized
    if isinstance(ds, T.TypeVar):
        raise TypeError(('Only a measure can appear on the last '
                        'position of a datashape, not %s') % repr(ds))
    return ds

if __name__ == '__main__':
    import readline
    # build the parse tablr
    rebuild()

    if len(sys.argv) > 1:
        ds_mod = open(sys.argv[1]).read()
        ast = list(parse_mod(ds_mod))
        print(ast)
    else:
        readline.parse_and_bind('')
        while True:
            try:
                line = raw_input('>> ')
                ast = parse(line)
                print(ast)
            except EOFError:
                break
