"""
Verify the validity of  IR.
"""

from __future__ import print_function, division, absolute_import
import functools

from .types import (Boolean, Integral, Real, Array, Struct, Pointer,
                    Vector, resolve_typedef)
from .ir import Function, Block, Value, Operation, Constant
from .traversal import visit, combine
from . import ops
from .pattern import match
from .utils import findallops

#===------------------------------------------------------------------===
# Utils
#===------------------------------------------------------------------===

class VerifyError(Exception):
    """Raised when we fail to verify IR"""

def unique(items):
    """Assert uniqueness of items"""
    seen = set()
    for item in items:
        if item in seen:
            raise VerifyError("Item not unique", item)
        seen.add(item)

#===------------------------------------------------------------------===
# Entry points
#===------------------------------------------------------------------===

@match
def verify(value, env=None):
    if isinstance(value, Function):
        verify_function(value)
    elif isinstance(value, Block):
        verify_operations(value)
    elif isinstance(value, Operation):
        verify_operation(value)
    else:
        assert isinstance(value, Value)

    return value, env

def op_verifier(func):
    """Verifying decorator for functions return a new (list of) Op"""
    @functools.wraps(func)
    def wrapper(*a, **kw):
        op = func(*a, **kw)
        if not isinstance(op, list):
            op = [op]
        for op in op:
            verify_op_syntax(op)
        return op

    return wrapper

#===------------------------------------------------------------------===
# Internal verification
#===------------------------------------------------------------------===

def verify_module(mod):
    """Verify a pykit module"""
    assert not set.intersection(set(mod.functions), set(mod.globals))
    for function in mod.functions.itervalues():
        verify_function(function)

def verify_function(func):
    try:
        _verify_function(func)
    except Exception as e:
        raise VerifyError("Error verifying function %s: %s" % (func.name, e))

def _verify_function(func):
    """Verify a pykit function"""
    # Verify arguments
    assert len(func.args) == len(func.type.argtypes)

    # Verify return presence and type
    restype = func.type.restype
    if not restype.is_void and not restype.is_opaque:
        rets = findallops(func, 'ret')
        for ret in rets:
            arg, = ret.args
            assert arg.type == restype, (arg.type, restype)

    verify_uniqueness(func)
    verify_block_order(func)
    verify_operations(func)
    verify_uses(func)
    verify_semantics(func)

def verify_uniqueness(func):
    """Verify uniqueness of register names and labels"""
    unique(block.name for block in func.blocks)
    unique(op for block in func.blocks for op in block)
    unique(op.result for block in func.blocks for op in block)

def verify_block_order(func):
    """Verify block order according to dominator tree"""
    from pykit.analysis import cfa

    flow = cfa.cfg(func)
    dominators = cfa.compute_dominators(func, flow)

    visited = set()
    for block in func.blocks:
        visited.add(block.name)
        for dominator in dominators[block.name]:
            if dominator not in visited:
                raise VerifyError("Dominator %s does not precede block %s" % (
                                                        dominator, block.name))

def verify_operations(func_or_block):
    """Verify all operations in the function or block"""
    for op in func_or_block.ops:
        verify_operation(op)

def verify_operation(op):
    """Verify a single Op"""
    assert op.block is not None, op
    assert op.result is not None, op
    verify_op_syntax(op)

def verify_op_syntax(op):
    """
    Verify the syntactic structure of the Op (arity, List/Value/Const, etc)
    """
    if op.opcode not in ops.op_syntax:
        return

    syntax = ops.op_syntax[op.opcode]
    vararg = syntax and syntax[-1] == ops.Star
    args = op.args
    if vararg:
        syntax = syntax[:-1]
        args = args[:len(syntax)]

    assert len(syntax) == len(args), (op, syntax)
    for arg, expected in zip(args, syntax):
        msg = (op, arg)
        if expected == ops.List:
            assert isinstance(arg, list), msg
        elif expected == ops.Const:
            assert isinstance(arg, Constant), msg
        elif expected == ops.Value:
            if op.opcode == "alloca":
                assert arg is None or isinstance(arg, Value), msg
            else:
                assert isinstance(arg, Value), msg
        elif expected == ops.Any:
            assert isinstance(arg, (Value, list)), msg
        elif expected == ops.Obj:
            pass
        else:
            raise ValueError("Invalid meta-syntax?", msg, expected)

def verify_uses(func):
    """Verify the def-use chains"""
    # NOTE: verify should be importable from any pass!
    from pykit.analysis import defuse
    uses = defuse.defuse(func)
    diff = set.difference(set(uses), set(func.uses))
    assert not diff, diff
    # assert uses == func.uses, (uses, func.uses)

# ______________________________________________________________________

class Verifier(object):
    """Semantic verification of all operations"""

def verify_semantics(func, env=None):
    verifier = combine(Verifier(), env and env.get("verify.handlers"))
    visit(verifier, func)

# ______________________________________________________________________

class LowLevelVerifier(object):

    def op_unary(self, op):
        assert type(op.type) in (Integral, Real)

    def op_binary(self, op):
        assert type(op.type) in (Integral, Real)

    def op_compare(self, op):
        assert type(op.type) in (Boolean,)
        left, right = op.args
        assert left.type == right.type
        assert type(left.type) in (Boolean, Integral, Real)

    def op_getfield(self, op):
        struct, attr = op.args
        assert struct.type.is_struct

    def op_setfield(self, op):
        struct, attr, value = op.args
        assert struct.type.is_struct


def verify_lowlevel(func):
    """
    Assert that the function is lowered for code generation.
    """
    for op in func.ops:
        assert type(resolve_typedef(op.type)) in (
            Boolean, Array, Integral, Real, Struct, Pointer, Function, Vector), op
