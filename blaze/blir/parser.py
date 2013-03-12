from ply import yacc
from errors import error

from lexer import tokens
from syntax import *

precedence = (
    ('left', 'LOR'),
    ('left', 'LAND'),
    ('nonassoc', 'LT','LE','GT','GE','EQ','NE'),
    ('left','PLUS','MINUS'),
    ('left','TIMES','DIVIDE'),
    ('right','UNARY'),
)

#------------------------------------------------------------------------

def p_mod(p):
    '''
    mod : body
    '''
    p[0] = Module(p[1], lineno=p.lineno(1))

#------------------------------------------------------------------------

def p_body(p):
    '''
    body : stmts
    '''
    p[0] = Statements(p[1], lineno=p.lineno(1))

def p_body_empty(p):
    '''
    body : empty
    '''
    p[0] = None

def p_stmts(p):
    '''
    stmts : stmts stmts
    '''
    p[0] = p[1] + p[2]

def p_stmt_single(p):
    '''
    stmts : stmt
    '''
    p[0] = [p[1]]

def p_stmt_decl(p):
    '''
    stmt : const_decl
         | var_decl
         | func_decl
         | foreign_decl
    '''
    p[0] = p[1]

def p_stmt_logic(p):
    '''
    stmt : assign_stmt
         | if_stmt
         | call_stmt
         | while_stmt
         | range_stmt
         | for_stmt
         | return_stmt
         | print_stmt
    '''
    p[0] = p[1]

#------------------------------------------------------------------------

def p_const_decl(p):
    '''
    const_decl : CONST ID ASSIGN expr SEMI
    '''
    p[0] = ConstDecl(p[2],p[4], lineno=p.lineno(1))

def p_var_decl(p):
    '''
    var_decl : VAR type ID SEMI
    '''
    p[0] = VarDecl(p[2],p[3], None, lineno=p.lineno(1))

def p_var_decl_expr(p):
    '''
    var_decl : VAR type ID ASSIGN expr SEMI
    '''
    p[0] = VarDecl(p[2],p[3],p[5], lineno=p.lineno(1))

def p_func_decl(p):
    '''
    func_decl : signature LBRACE body RBRACE
    '''
    p[0] = FunctionDef(p[1], p[3], lineno=p.lineno(2))

def p_foreign_decl(p):
    '''
    foreign_decl : FOREIGN STRING signature SEMI
    '''
    p[0] = ExternFuncDecl(p[2], p[3], lineno=p.lineno(1))

def p_signature(p):
    '''
    signature : DEF ID LPAREN params RPAREN ARROW type
    '''
    p[0] = FunctionSig(p[2], p[4], p[7], lineno=p.lineno(1))

def p_signature_empty(p):
    '''
    signature : DEF ID LPAREN RPAREN ARROW type
    '''
    p[0] = FunctionSig(p[2], [], p[6], lineno=p.lineno(1))

def p_signature_void(p):
    '''
    signature : DEF ID LPAREN RPAREN empty
    '''
    p[0] = FunctionSig(p[2], [], Type('void'), lineno=p.lineno(1))

#------------------------------------------------------------------------

def p_params(p):
    '''
    params : params COMMA param
    '''
    p[0] = p[1]
    p[0].append(p[3])

def p_params_single(p):
    '''
    params : param
    '''
    p[0] = [p[1]]

def p_param_decl(p):
    '''
    param : ID COLON type
    '''
    p[0] = ParmDeclaration(p[1], p[3], lineno=p.lineno(1))

#------------------------------------------------------------------------

def p_assign_stmt(p):
    '''
    assign_stmt : store_location ASSIGN expr SEMI
    '''
    p[0] = Assign(p[1], p[3], lineno=p.lineno(2))

#------------------------------------------------------------------------

def p_print_stmt(p):
    '''
    print_stmt : PRINT expr SEMI
    '''
    p[0] = Print(p[2],lineno=p.lineno(1))

def p_range_stmt(p):
    '''
    range_stmt : RANGE LPAREN exprlist RPAREN
    '''
    bounds = p[3]
    if len(bounds) == 2:
        start = p[3][0]
        stop = p[3][1]
        p[0] = Range(start, stop, lineno=p.lineno(1))
    else:
        start = Const(0)
        stop = p[3][0]
        p[0] = Range(start, stop, lineno=p.lineno(1))

def p_if_stmt(p):
    '''
    if_stmt : IF expr LBRACE body RBRACE
    '''
    p[0] = IfElseStatement(p[2], p[4], None, lineno=p.lineno(1))

def p_ifelse_stmt(p):
    '''
    if_stmt : IF expr LBRACE body RBRACE ELSE LBRACE body RBRACE
    '''
    p[0] = IfElseStatement(p[2], p[4], p[8], lineno=p.lineno(1))

def p_while_stmt(p):
    '''
    while_stmt : WHILE expr LBRACE body RBRACE
    '''
    p[0] = WhileStatement(p[2], p[4], lineno=p.lineno(1))

def p_for_stmt(p):
    '''
    for_stmt : FOR ID IN range_stmt LBRACE body RBRACE
    '''
    p[0] = ForStatement(p[2], p[4], p[6], lineno=p.lineno(1))

def p_return_stmt(p):
    '''
    return_stmt : RETURN expr SEMI
    '''
    p[0] = ReturnStatement(p[2], lineno=p.lineno(1))

#------------------------------------------------------------------------

def p_expr_unary(p):
    '''
    expr : PLUS  expr %prec UNARY
         | MINUS expr %prec UNARY
         | LNOT  expr %prec UNARY
    '''
    p[0] = UnaryOp(p[1],p[2], lineno=p.lineno(1))

def p_expr_binary(p):
    '''
    expr : expr PLUS expr
         | expr MINUS expr
         | expr TIMES expr
         | expr DIVIDE expr
    '''
    p[0] = BinOp(p[2],p[1],p[3], lineno=p.lineno(2))

def p_relational_expr_binary(p):
    '''
    expr : expr LT expr
         | expr LE expr
         | expr GT expr
         | expr GE expr
         | expr EQ expr
         | expr NE expr
         | expr LOR expr
         | expr LAND expr
    '''
    p[0] = Compare(p[2], p[1], p[3], lineno=p.lineno(2))

#------------------------------------------------------------------------

def p_expr_group(p):
    '''
    expr : LPAREN expr RPAREN
    '''
    p[0] = p[2]

def p_expr_func_call(p):
    '''expr : call_expr'''
    p[0] = p[1]

def p_expr_location(p):
    '''
    expr : load_location
    '''
    p[0] = p[1]

def p_expr_literal(p):
    '''
    expr : literal
    '''
    p[0] = p[1]

def p_exprlist(p):
    '''
    exprlist : exprlist COMMA exprlist
    '''
    p[0] = p[1] + p[3]

def p_exprlist_single(p):
    '''
    exprlist : expr
    '''
    p[0] = [p[1]]

#------------------------------------------------------------------------

def p_call_stmt(p):
    '''
    call_stmt : call_expr SEMI
    '''
    p[0] = p[1]

def p_function_call1(p):
    '''
    call_expr : ID LPAREN exprlist RPAREN
    '''
    p[0] = FunctionCall(p[1], p[3], lineno=p.lineno(1))

def p_function_call2(p):
    '''
    call_expr : ID LPAREN RPAREN
    '''
    p[0] = FunctionCall(p[1], [], lineno=p.lineno(1))

#------------------------------------------------------------------------

def p_literal(p):
    '''
    literal : INTEGER
            | FLOAT
            | STRING
    '''
    p[0] = Const(p[1], lineno=p.lineno(1))

def p_literal_bool(p):
    '''
    literal : TRUE
            | FALSE
    '''
    if p[1] == 'True':
        val = True
    else:
        val = False
    p[0] = Const(val, lineno=p.lineno(1))

#------------------------------------------------------------------------

def p_load_location_var(p):
    '''
    load_location : ID
    '''
    p[0] = LoadVariable(p[1], lineno=p.lineno(1))

def p_store_location_var(p):
    '''
    store_location : ID
    '''
    p[0] = StoreVariable(p[1], lineno=p.lineno(1))

#------------------------------------------------------------------------

def p_tuple1(p):
    '''
    tuple : tuple COMMA tuple
    '''
    p[0] = Tuple(p[1] + p[3])

def p_tuple2(p):
    '''
    tuple : expr
    '''
    p[0] = [p[1]]

#------------------------------------------------------------------------

def p_load_location_index(p):
    '''
    load_location : ID LBRACKET tuple RBRACKET
                  | ID LBRACKET expr RBRACKET
    '''
    p[0] = LoadIndex(p[1], p[3], lineno=p.lineno(1))

def p_store_location_index(p):
    '''
    store_location : ID LBRACKET tuple RBRACKET
                   | ID LBRACKET expr RBRACKET
    '''
    p[0] = StoreIndex(p[1], p[3], lineno=p.lineno(1))

#------------------------------------------------------------------------

def p_paramtype(p):
    '''
    type : type LBRACKET type RBRACKET
    '''
    p[0] = ParamType(p[1], p[3], lineno=p.lineno(1))

def p_typename(p):
    '''
    type : ID
    '''
    p[0] = Type(p[1], lineno=p.lineno(1))

#------------------------------------------------------------------------

def p_empty(p):
    '''
    empty :
    '''
    pass

#------------------------------------------------------------------------

def p_error(p):
    if p:
        error(p.lineno, "Syntax error in input at token '%s'" % p.value)
    else:
        error("EOF", "Syntax error. No more input.")

#------------------------------------------------------------------------
# Toplevel
#------------------------------------------------------------------------

def make_parser(debug=False):
    import sys
    import blex
    import byacc
    import lexer
    from plyhacks import lexfrom, yaccfrom
    from functools import partial

    module = sys.modules[__name__]
    lexer = lexfrom(lexer, blex)
    parser = yaccfrom(module, byacc, lexer)

    return partial(parser.parse, lexer=lexer)

#------------------------------------------------------------------------
# --ddump-parse
#------------------------------------------------------------------------

def ddump_parse(source):
    import sys
    import lexer
    import errors
    from astutils import dump

    lexer = lexer.make_lexer()
    parser = yacc.yacc()

    with errors.listen():
        program = parser.parse(source)

    sys.stdout.write(dump(program)+'\n')

#------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    source = open(sys.argv[1]).read()
    ddump_parse(source)
