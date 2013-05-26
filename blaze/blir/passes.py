from . import cfg
from . import exc
from . import parser
from . import errors
from . import codegen
from . import typecheck
from . import optimizations

import sys
import time
import argparse
from threading import Lock

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
                raise CompileError(ppass.__name__)
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
    ast = parser.parse(ast)
    return ast, env

# ------------------------------

@ppass("Type checker")
def typecheck_pass(ast, env):
    from . import btypes
    symtab = typecheck.typecheck(ast, typesystem=btypes)

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
    symtab = env['symtab']

    cgen = codegen.LLVMEmitter(symtab)
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
    args = env['args']

    opt_level = args['O']
    llmodule = cgen.module

    llmodule.verify()

    # module-level optimization
    if not args.get('nooptimize', None):
        novectorize = args.get('novectorize', False)

        optimizer = optimizations.LLVMOptimizer(
            llmodule,
            opt_level,
            loop_vectorize=(not novectorize),
        )

    llmodule.verify()
    env['lmodule'] = llmodule

    return ast, env

#------------------------------------------------------------------------
# Pipeline
#------------------------------------------------------------------------

frontend = Pipeline('frontend', [parse_pass,
                                 typecheck_pass,
                                 rewrite_pass
                                 ])

backend = Pipeline('backend', [ssa_pass,
                               codegen_pass,
                               optimizer_pass,
                               ])

compiler = Pipeline('compile', [frontend,
                                backend
                                ])

#------------------------------------------------------------------------
# Toplevel
#------------------------------------------------------------------------

def compile(source, **opts):
    """
    Compile the given Blir source using

    Parameters
    ----------
    source : str
        Blir source.

    opts : dict
        Options for Blir compile pipeline.

         * 'O' : Optimization level.
         * 'nooptimize' : Don't run the LLVM optimization pass.
         * 'noprelude' : Don't link against the prelude.

    """
    if len(source) == '':
        raise ValueError("Empty source string")

    opts.setdefault('O', 2)
    env = {'args': opts}

    # llvm compilation is very not thread safe
    with compilelock:
        ast, env = compiler(source, env)
    return ast, env

#------------------------------------------------------------------------
# Command Line Interface
#------------------------------------------------------------------------

def main():
    argp = argparse.ArgumentParser('blirc')
    argp.add_argument('input', metavar="input", nargs='?', help='Source file')
    argp.add_argument('-O', metavar="opt", nargs='?', type=int, help='Optimization level', default=2)
    argp.add_argument('--ddump-parse', action='store_true', help='Dump parse tree')
    argp.add_argument('--ddump-lex', action='store_true', help='Dump token stream')
    argp.add_argument('--ddump-blocks', action='store_true', help='Dump the block structure')
    argp.add_argument('--ddump-tc', action='store_true', help='Dump the type checker state')
    argp.add_argument('--noprelude', action='store_true', help='Don\'t link against the prelude')
    argp.add_argument('--nooptimize', action='store_true', help='Don\'t run LLVM optimization pass')
    argp.add_argument('--novectorize', action='store_true', help='Enable SIMD loop vectorization')
    argp.add_argument('--emit-llvm', '-S', action='store_true', help=' Generate output files in LLVM formats ')
    argp.add_argument('--emit-x86', action='store_true', help=' Generate output files in x86 assembly ')
    argp.add_argument('--run', action='store_true', help='Execute main() in generated code ')
    args = argp.parse_args()

    if args.input:
        source = open(args.input).read()
    else:
        sys.stderr.write('No input\n')
        sys.exit(1)

    if args.ddump_lex:
        parser.ddump_lex(source)

    if args.ddump_parse:
        parser.ddump_parse(source)

    if args.ddump_blocks:
        cfg.ddump_blocks(source)

    try:
        # =====================================
        start = time.time()
        with errors.listen():
            opts = dict(args._get_kwargs())
            ast, env = compile(source, **opts)
        timing = time.time() - start
        # =====================================

        if args.emit_llvm:
            print(env['lmodule'])
        elif args.emit_x86:
            print(env['lmodule'].to_native_assembly())
        elif args.run:
            ctx = exc.Context(env)
            exc.execute(ctx, fname='main')
        else:
            print('Compile time %.3fs' % timing)

    except CompileError as e:
        sys.stderr.write('FAIL: Failure in compiler phase: %s\n' % e.args[0])
        errors.reset()
        sys.exit(1)

if __name__ == '__main__':
    main()
