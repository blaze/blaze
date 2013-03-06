import sys
import time

import lexer
import parser

import cfg
import typecheck
import codegen
import errors
import exc

from threading import Lock
from plyhacks import lexfrom, yaccfrom

compilelock = Lock()

#------------------------------------------------------------------------
# Pipeline
#------------------------------------------------------------------------

class CompileError(RuntimeError):
    pass

class Pipeline(object):

    def __init__(self, name, passes):
        self.name = name
        self.__name__ = name
        self.passes = passes

    def __call__(self, ast, env):

        for ppass in self.passes:
            ast, env = ppass(ast, env)
            if errors.occurred():
                errors.reset()
                raise CompileError, ppass.__name__
        return ast, env

#------------------------------------------------------------------------
# Passes
#------------------------------------------------------------------------

def ppass(name):
    def wrapper(fn):
        fn.__name__ = name
        return fn
    return wrapper

# ------------------------------

@ppass("Syntax Parser")
def parse_pass(ast, env):
    parse = parser.make_parser()

    ast = parse(ast)
    return ast, env

# ------------------------------

@ppass("Type checker")
def typecheck_pass(ast, env):
    symtab = typecheck.typecheck(ast)

    env['symtab'] = symtab

    return ast, env

# ------------------------------

@ppass("Rewriter")
def rewrite_pass(ast, env):
    return ast, env

# ------------------------------

@ppass("Single static assignment")
def ssa_pass(ast, env):
    functions = cfg.ssa_pass(ast)

    env['functions'] = functions

    return ast, env

# ------------------------------

@ppass("Code generation")
def codegen_pass(ast, env):
    cgen = codegen.LLVMEmitter()
    blockgen = codegen.BlockEmitter(cgen)

    env['cgen'] = cgen
    env['blockgen'] = blockgen

    functions = env['functions']
    lfunctions = []

    for name, retty, argtys, start_block in functions:
        function = blockgen.generate_function(
            name,
            retty,
            argtys,
            start_block
        )
        function.verify()
        lfunctions.append(function)

    env['lfunctions'] = lfunctions

    return ast, env

# ------------------------------

@ppass("LLVM Optimizer")
def optimizer_pass(ast, env):
    cgen = env['cgen']
    lfunctions = env['lfunctions']

    opt_level = env['args']['O']
    optimizer = codegen.LLVMOptimizer(cgen.module, opt_level)

    # function-level optimize
    for lfunc in lfunctions:
        optimizer.run(lfunc)
        lfunc.verify()

    # module-level optimization
    optimizer.runmodule(cgen.module)

    cgen.module.verify()
    env['lmodule'] = cgen.module

    return ast, env

# ------------------------------

@ppass("Linker")
def linker_pass(ast, env):
    return ast, env

#------------------------------------------------------------------------
# Pipeline Structure
#------------------------------------------------------------------------

frontend = Pipeline('frontend', [parse_pass,
                                 typecheck_pass,
                                 rewrite_pass
                                 ])

backend = Pipeline('backend', [ssa_pass,
                               codegen_pass,
                               optimizer_pass,
                               linker_pass,
                               ])

compiler = Pipeline('compile', [frontend,
                                backend
                                ])

#------------------------------------------------------------------------
# Toplevel
#------------------------------------------------------------------------

def compile(source, opts=None):
    env = {'args': opts}
    with compilelock:
        ast, env = compiler(source, env)
    return ast, env

#------------------------------------------------------------------------
# Command Line Interface
#------------------------------------------------------------------------

def main():
    import argparse
    argp = argparse.ArgumentParser('blirc')
    argp.add_argument('file', metavar="file", nargs='?', help='Source file')
    argp.add_argument('-O', metavar="opt", nargs='?', type=int, help='Optimization level', default=2)
    argp.add_argument('--ddump-parse', action='store_true', help='Dump parse tree')
    argp.add_argument('--ddump-lex', action='store_true', help='Dump token stream')
    argp.add_argument('--ddump-blocks', action='store_true', help='Dump the block structure')
    argp.add_argument('--ddump-tc', action='store_true', help='Dump the type checker state')
    argp.add_argument('--ddump-optimizer', action='store_true', help='Dump diff of the LLVM optimizer pass')
    argp.add_argument('--noprelude', action='store_true', help='Don\'t link against the prelude')
    argp.add_argument('--nooptimize', action='store_true', help='Don\'t run LLVM optimization pass')
    argp.add_argument('--emit-llvm', action='store_true', help=' Generate output files in LLVM formats ')
    argp.add_argument('--emit-x86', action='store_true', help=' Generate output files in x86 assembly ')
    argp.add_argument('--run', action='store_true', help='Execute generated code ')
    args = argp.parse_args()

    if args.file:
        source = open(args.file).read()
    else:
        sys.stderr.write('No input\n')
        sys.exit(1)

    if args.ddump_lex:
        lexer.ddump_lex(source)

    if args.ddump_parse:
        parser.ddump_parse(source)

    if args.ddump_blocks:
        cfg.ddump_blocks(source)

    if args.ddump_optimizer:
        codegen.ddump_optimizer(source)

    if args.ddump_tc:
        typecheck.ddump_tc(source)

    try:
        # =====================================
        start = time.time()
        with errors.listen():
            opts = vars(args)
            ast, env = compile(source, opts)
        timing = time.time() - start
        # =====================================

        if args.emit_llvm:
            print env['lmodule']
        elif args.emit_x86:
            print env['lmodule'].to_native_assembly()
        elif args.run:
            exc.execute(env)
        else:
            print 'Compile time %.3fs' % timing

    except CompileError as e:
        sys.stderr.write('FAIL: Failure in compiler phase: %s\n' % e.args[0])
        sys.exit(1)

if __name__ == '__main__':
    main()
