"""
LLVM module for compiling DataShape objects to ADTs, Using this
we generate the usual Sum & Product types as well as derivative
user-defined parametric types which all JIT down to types in the
execution.
"""

import ctypes
from string import letters
from collections import namedtuple
from contextlib import contextmanager

from llvm.core import Module, Constant, Type, Function, Builder, \
    Type, GlobalValue, GlobalVariable
from llvm.ee import ExecutionEngine

# +------------------------------------+--------------------------------------+
# | inline struct Maybe __Nothing__() {| inline struct Maybe __Just__(int a) {|
# |   struct Maybe x = {Nothing};      |   struct Maybe x = {Just};           |
# |   return x;                        |   x.Just.a = a;                      |
# | }                                  |   return x;                          |
# |                                    | }                                    |
# +------------------------------------+--------------------------------------+
#                   ^                                      ^
#                   |         +----------------------------+
#                   |         |
# Maybe a = Nothing | Just a
#    |
#    v
# +----------------------------+
# | struct Maybe {             |
# |   enum {Just,Nothing} tag; |
# |   union {                  |
# |     struct {               |
# |       int a;               |
# |     } Just;                |
# |     struct {               |
# |     } Nothing;             |
# |   };                       |
# | };                         |
# +----------------------------+
#

#------------------------------------------------------------------------
# Definitions
#------------------------------------------------------------------------

void      = Type.void()
char      = Type.int(8)
short     = Type.int(16)
int       = Type.int(32)
int64     = Type.int(64)
float     = Type.float()
double    = Type.double()
int8      = Type.int(8)
int8p     = Type.pointer(int8)

# fountain of free variables
free = lambda: iter(letters)

def const(n):
    return Constant.int(int, n)

spine = namedtuple('spine', 'name, params, values')
value = namedtuple('value', 'name, params')

#------------------------------------------------------------------------
# Instances
#------------------------------------------------------------------------

# TypeVar, not unique
AnyT = spine( 'Any', [], [] )

# TypeVar, unique within defition
TypeVarT = spine( 'TypeVar', ['id'], ['TypeVar', ['id']] )

# Fixed Dimension
FixedT = spine(
    'Fixed', ['n'],
    [
        value('Fixed', ['n'])
    ]
)

MaybeT = spine(
    'Maybe', ['a'],
    [
        value('Just', ['a']),
        value('Nothing', []),
    ]
)

EitherT = spine(
    'Either', ['a', 'b'],
    [
        value('Left', ['a']),
        value('Right', ['b']),
    ]
)

#------------------------------------------------------------------------
# Construction
#------------------------------------------------------------------------

def build_constructor(fn, ty, spine, j):
    freev = free()

    entry = fn.append_basic_block('entry')

    builder = Builder.new(entry)
    retval = builder.alloca(spine, name='ret')

    #tag = builder.gep(retval, [const(0), const(0)], 'tag')
    #builder.store(const(j), tag)

    for i, arg in enumerate(fn.args):
        idx1 = builder.gep(retval, [const(0), const(0)], next(freev))
        idx2 = builder.gep(idx1, [const(0), const(i)], next(freev))
        builder.store(const(25), idx2)
    return builder.ret(retval)

def create_instance(mod, spec, ctx):

    #  Spine
    #    |
    #
    # T ... = A .. | B .. | C ..
    #
    #                |
    #              Values

    lvals = []
    instances = []

    # Values
    # ======
    for value in spec.values:
        tys = [ctx[id] for id in value.params]
        lvals += [(value.name, Type.struct(tys, value.name))]

    # Spine
    # ======
    spine = Type.struct([a[1] for a in lvals], 'maybe')

    for i, (name, value) in enumerate(lvals):
        fn_spec = Type.function(void, value.elements)
        F = mod.add_function(fn_spec, value.name)
        instances += [F]

        build_constructor(F, value, spine, 1)

    return spine, instances

#------------------------------------------------------------------------
# Debug
#------------------------------------------------------------------------

def wrap_constructor(func, engine, py_module ,rettype):
    from bitey.bind import map_llvm_to_ctypes
    args = func.type.pointee.args
    ret_type = func.type.pointee.return_type
    ret_ctype = map_llvm_to_ctypes(ret_type, py_module)
    args_ctypes = [map_llvm_to_ctypes(arg, py_module) for arg in args]

    functype = ctypes.CFUNCTYPE(rettype, *args_ctypes)
    addr = engine.get_pointer_to_function(func)
    return functype(addr)

def debug_ctypes(mod, spine, constructors):
    from bitey.bind import map_llvm_to_ctypes
    from imp import new_module
    engine = ExecutionEngine.new(mod)

    py = new_module('')
    map_llvm_to_ctypes(spine, py)

    return [wrap_constructor(c, engine, py, py.maybe) for c in constructors]

if __name__ == '__main__':
    module = Module.new('adt')
    module.add_library("c")

    spine, values = create_instance(module, MaybeT, {'a': int, 'b': int})
    a,b = debug_ctypes(module, spine, values)
