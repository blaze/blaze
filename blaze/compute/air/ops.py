from __future__ import print_function, division, absolute_import
import collections

try:
    intern
except NameError:
    intern = lambda s: s

#===------------------------------------------------------------------===
# Syntax
#===------------------------------------------------------------------===

all_ops = []
op_syntax = {} # Op -> Syntax

List  = collections.namedtuple('List',  []) # syntactic list
Value = collections.namedtuple('Value', []) # single Value
Const = collections.namedtuple('Const', []) # syntactic constant
Any   = collections.namedtuple('Any',   []) # Value | List
Star  = collections.namedtuple('Star',  []) # any following arguments
Obj   = collections.namedtuple('Obj',   []) # any object

fmts = {'l': List, 'v': Value, 'c': Const, 'a': Any, '*': Star, 'o': Obj}

# E.g. op('foo', List, Const, Value, Star) specificies an opcode 'foo' accepting
# as the argument list a list of arguments, a constant, an operation and any
# trailing arguments. E.g. [[], Const(...), Op(...)] would be valid.

def op(name, *args):
    if '/' in name:
        name, fmt = name.split('/')
        args = [fmts[c] for c in fmt]

    name = intern(name)
    all_ops.append(name)
    op_syntax[name] = list(args)
    return name

#===------------------------------------------------------------------===
# Typed IR (initial input)
#===------------------------------------------------------------------===

# IR Constants. Constants start with an uppercase letter

# math
Sin                = 'Sin'
Asin               = 'Asin'
Sinh               = 'Sinh'
Asinh              = 'Asinh'
Cos                = 'Cos'
Acos               = 'Acos'
Cosh               = 'Cosh'
Acosh              = 'Acosh'
Tan                = 'Tan'
Atan               = 'Atan'
Atan2              = 'Atan2'
Tanh               = 'Tanh'
Atanh              = 'Atanh'
Log                = 'Log'
Log2               = 'Log2'
Log10              = 'Log10'
Log1p              = 'Log1p'
Exp                = 'Exp'
Exp2               = 'Exp2'
Expm1              = 'Expm1'
Floor              = 'Floor'
Ceil               = 'Ceil'
Abs                = 'Abs'
Erfc               = 'Erfc'
Rint               = 'Rint'
Pow                = 'Pow'
Round              = 'Round'

# ______________________________________________________________________
# Constants

constant           = op('constant/o')         # object pyval

# ______________________________________________________________________
# Variables

alloca             = op('alloca/o')           # obj numItems [length of allocation implied by return type]
load               = op('load/v')             # alloc var
store              = op('store/vv')           # expr value, alloc var

# ______________________________________________________________________
# Conversion

convert            = op('convert/v')          # expr arg
bitcast            = op('bitcast/v')          # expr value

# ______________________________________________________________________
# Control flow

# Basic block leaders
phi                = op('phi/ll')             # expr *blocks, expr *values
exc_setup          = op('exc_setup/l')        # block *handlers
exc_catch          = op('exc_catch/l')        # expr *types

# Basic block terminators
jump               = op('jump/v')             # block target
cbranch            = op('cbranch/vvv')        # (expr test, block true_target,
                                              #  block false_target)
exc_throw          = op('exc_throw/v')        # expr exc
ret                = op('ret/o')              # expr result

# ______________________________________________________________________
# Functions

call               = op('call/vl')            # expr obj, expr *args
call_math          = op('call_math/ol')       # str name, expr *args

# ______________________________________________________________________
# sizeof

addressof          = op('addressof/v')        # expr obj
sizeof             = op('sizeof/v')           # expr obj

# ______________________________________________________________________
# Pointers

ptradd             = op('ptradd/vv')          # expr pointer, expr value
ptrload            = op('ptrload/v')          # expr pointer
ptrstore           = op('ptrstore/vv')        # expr value, expr pointer
ptrcast            = op('ptrcast/v')          # expr pointer
ptr_isnull         = op('ptr_isnull/v')       # expr pointer

# ______________________________________________________________________
# Unified: Structs/Arrays/Objects/Vectors

get                = op('get/vl')        # (expr value, list index)
set                = op('set/vvl')       # (expr value, expr value, list index)

# ______________________________________________________________________
# Attributes

getfield           = op('getfield/vo')        # (expr value, str attr)
setfield           = op('setfield/vov')       # (expr value, str attr, expr value)

# ______________________________________________________________________
# Fields

extractfield       = op('extractfield/vo')
insertfield        = op('insertfield/vov')

# ______________________________________________________________________
# Vectors

shufflevector      = op('shufflevector/vvv')  # (expr vector0, expr vector1, expr vector2)

# ______________________________________________________________________
# Basic operators

# Binary
add                = op('add/vv')
sub                = op('sub/vv')
mul                = op('mul/vv')
div                = op('div/vv')
mod                = op('mod/vv')
lshift             = op('lshift/vv')
rshift             = op('rshift/vv')
bitand             = op('bitand/vv')
bitor              = op('bitor/vv')
bitxor             = op('bitxor/vv')

# Unary
invert             = op('invert/v')
not_               = op('not_/v')
uadd               = op('uadd/v')
usub               = op('usub/v')

# Compare
eq                 = op('eq/vv')
ne                 = op('ne/vv')
lt                 = op('lt/vv')
le                 = op('le/vv')
gt                 = op('gt/vv')
ge                 = op('ge/vv')
is_                = op('is_/vv')

# ______________________________________________________________________
# Exceptions

check_error        = op('check_error/vv')   # (expr arg, expr badval)
new_exc            = op('new_exc/vv')       # (expr exc, expr? msg)

# ______________________________________________________________________
# Debugging

print              = op('print/v')

# ______________________________________________________________________
# Opcode utils

import fnmatch

void_ops = (print, store, ptrstore, exc_setup, exc_catch, jump, cbranch, exc_throw,
            ret, setfield, check_error)

is_leader     = lambda x: x in (phi, exc_setup, exc_catch)
is_terminator = lambda x: x in (jump, cbranch, exc_throw, ret)
is_void       = lambda x: x in void_ops

def oplist(pattern):
    """Given a pattern, return all matching opcodes, e.g. thread_*"""
    for name, value in globals().iteritems():
        if not name.startswith('__') and fnmatch.fnmatch(name, pattern):
            yield value
