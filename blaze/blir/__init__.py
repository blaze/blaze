from .passes import compile, CompileError
from .exc import Context, execute
from .errors import log

def bitcode(env):
    """ Print LLVM bitcode for the given compiled environment """
    return env['cgen'].module

def assembly(env):
    """ Print x86 assembly for the given compiled environment """
    return env['cgen'].module.to_native_assembly()

def module(env):
    """
    Extract the LLVM module from Blir environment.

    >>> from blir import compile
    >>> ast, env = compile(source)
    >>> module(env)
    <llvm.core.Module at 0x2205cd0>
    """
    return env['lmodule']

def test(verbosity=1, repeat=1):
    from blir import test_blir
    return test_blir.run(verbosity=verbosity, repeat=repeat)
