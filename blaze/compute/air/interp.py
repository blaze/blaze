"""IR interpreter."""

from __future__ import print_function, division, absolute_import

import ctypes

try:
    import exceptions
except ImportError:
    import builtins as exceptions
from itertools import chain
from collections import namedtuple
from functools import partial

from . import defs, ops, tracing, types
from .ir import Function
from .traversal import ArgLoader
from .utils import linearize
#===------------------------------------------------------------------===
# Interpreter
#===------------------------------------------------------------------===

Undef = "Undef"                         # Undefined/uninitialized value
State = namedtuple('State', ['refs'])   # State shared by stack frames

class Reference(object):
    """
    Models a reference to an object
    """

    def __init__(self, obj, refcount, producer):
        self.obj = obj
        self.refcount = refcount
        self.producer = producer

class UncaughtException(Exception):
    """
    Raised by the interpreter when code raises an exception that isn't caught
    """

class Interp(object):
    """
    Interpret the function given as a ir.Function. See the run() function
    below.

        func:           The ir.Function we interpret
        exc_model:      ExceptionModel that knows how to deal with exceptions
        argloader:      InterpArgloader: knows how pykit Values are associated
                        with runtime (stack) values (loads from the store)
        ops:            Flat list of instruction targets (['%0'])
        blockstarts:    Dict mapping block labels to address offsets
        prevblock:      Previously executing basic block
        pc:             Program Counter
        lastpc:         Last value of Program Counter
        exc_handlers:   List of exception target blocks to try
        exception:      Currently raised exception
        refs:           { id(obj) : Reference }
    """

    def __init__(self, func, env, exc_model, argloader, tracer):
        self.func = func
        self.env = env
        self.exc_model = exc_model
        self.argloader = argloader

        self.state = {
            'env':       env,
            'exc_model': exc_model,
            'tracer':    tracer,
        }

        self.ops, self.blockstarts = linearize(func)
        self.lastpc = 0
        self._pc = 0
        self.prevblock = None
        self.exc_handlers = None
        self.exception = None

    # __________________________________________________________________
    # Utils

    def incr_pc(self):
        """Increment program counter"""
        self.pc += 1

    def decr_pc(self):
        """Decrement program counter"""
        self.pc -= 1

    def halt(self):
        """Stop interpreting"""
        self.pc = -1

    @property
    def op(self):
        """Return the current operation"""
        return self.getop(self.pc)

    def getop(self, pc):
        """PC -> Op"""
        return self.ops[pc]

    def setpc(self, newpc):
        self.lastpc = self.pc
        self._pc = newpc

    pc = property(lambda self: self._pc, setpc, doc="Program Counter")

    def blockswitch(self, oldblock, newblock, valuemap):
        self.prevblock = oldblock
        self.exc_handlers = []

        self.execute_phis(newblock, valuemap)

    def execute_phis(self, block, valuemap):
        """
        Execute all phis in parallel, i.e. execute them before updating the
        store.
        """
        new_values = {}
        for op in block.leaders:
            if op.opcode == 'phi':
                new_values[op.result] = self.execute_phi(op)

        valuemap.update(new_values)

    def execute_phi(self, op):
        for i, block in enumerate(op.args[0]):
            if block == self.prevblock:
                values = op.args[1]
                return self.argloader.load_op(values[i])

        raise RuntimeError("Previous block %r not a predecessor of %r!" %
                                    (self.prevblock.name, op.block.name))

    noop = lambda *args: None

    # __________________________________________________________________
    # Core operations

    # unary, binary and compare operations set below

    def convert(self, arg):
        return types.convert(arg, self.op.type)

    # __________________________________________________________________
    # Var

    def alloca(self, numitems=None):
        return { 'value': Undef, 'type': self.op.type }

    def load(self, var):
        #assert var['value'] is not Undef, self.op
        return var['value']

    def store(self, value, var):
        if isinstance(value, dict) and set(value) == set(['type', 'value']):
            value = value['value']
        var['value'] = value

    def phi(self):
        "See execute_phis"
        return self.argloader.load_op(self.op)

    # __________________________________________________________________
    # Functions

    def function(self, funcname):
        return self.func.module.get_function(funcname)

    def call(self, func, args):
        if isinstance(func, Function):
            # We're calling another known pykit function,
            try:
                return run(func, args=args, **self.state)
            except UncaughtException as e:
                # make sure to handle any uncaught exceptions properly
                self.exception, = e.args
                self._propagate_exc()
        else:
            return func(*args)

    def call_math(self, fname, *args):
        return defs.math_funcs[fname](*args)

    # __________________________________________________________________
    # Attributes

    def getfield(self, obj, attr):
        if obj['value'] is Undef:
            return Undef
        return obj['value'][attr] # structs are dicts

    def setfield(self, obj, attr, value):
        if obj['value'] is Undef:
            obj['value'] = {}
        obj['value'][attr] = value

    # __________________________________________________________________

    print = print

    # __________________________________________________________________
    # Pointer

    def ptradd(self, ptr, addend):
        value = ctypes.cast(ptr, ctypes.c_void_p).value
        itemsize = ctypes.sizeof(type(ptr)._type_)
        return ctypes.cast(value + itemsize * addend, type(ptr))

    def ptrload(self, ptr):
        return ptr[0]

    def ptrstore(self, value, ptr):
        ptr[0] = value

    def ptr_isnull(self, ptr):
        return ctypes.cast(ptr, ctypes.c_void_p).value == 0

    def func_from_addr(self, ptr):
        type = self.op.type
        return ctypes.cast(ptr, types.to_ctypes(type))

    # __________________________________________________________________
    # Control flow

    def ret(self, arg):
        self.halt()
        if self.func.type.restype != types.Void:
            return arg

    def cbranch(self, test, true, false):
        if test:
            self.pc = self.blockstarts[true.name]
        else:
            self.pc = self.blockstarts[false.name]

    def jump(self, block):
        self.pc = self.blockstarts[block.name]

    # __________________________________________________________________
    # Exceptions

    def new_exc(self, exc_name, exc_args):
        return self.exc_model.exc_instantiate(exc_name, *exc_args)

    def exc_catch(self, types):
        self.exception = None # We caught it!

    def exc_setup(self, exc_handlers):
        self.exc_handlers = exc_handlers

    def exc_throw(self, exc):
        self.exception = exc
        self._propagate_exc() # Find exception handler

    def _exc_match(self, exc_types):
        """
        See whether the current exception matches any of the exception types
        """
        return any(self.exc_model.exc_match(self.exception, exc_type)
                        for exc_type in exc_types)

    def _propagate_exc(self):
        """Propagate installed exception (`self.exception`)"""
        catch_op = self._find_handler()
        if catch_op:
            # Exception caught! Transfer control to block
            catch_block = catch_op.parent
            self.pc = self.blockstarts[catch_block.name]
        else:
            # No exception handler!
            raise UncaughtException(self.exception)

    def _find_handler(self):
        """Find a handler for an active exception"""
        exc = self.exception

        for block in self.exc_handlers:
            for leader in block.leaders:
                if leader.opcode != ops.exc_catch:
                    continue

                args = [arg.const for arg in leader.args[0]]
                if self._exc_match(args):
                    return leader

    # __________________________________________________________________
    # Generators

    def yieldfrom(self, op):
        pass # TODO:

    def yieldval(self, op):
        pass # TODO:


# Set unary, binary and compare operators
for opname, evaluator in chain(defs.unary.items(), defs.binary.items(),
                               defs.compare.items()):
    setattr(Interp, opname, staticmethod(evaluator))

#===------------------------------------------------------------------===
# Exceptions
#===------------------------------------------------------------------===

class ExceptionModel(object):
    """
    Model that governs the exception hierarchy
    """

    def exc_op_match(self, exc_type, op):
        """
        See whether `exception` matches `exc_type`
        """
        assert exc_type.opcode == 'constant'
        if op.opcode == 'constant':
            return self.exc_match(exc_type.const, op.const)
        raise NotImplementedError("Dynamic exception checks")

    def exc_match(self, exc_type, exception):
        """
        See whether `exception` matches `exc_type`
        """
        return (isinstance(exc_type, exception) or
                issubclass(exception, exc_type))

    def exc_instantiate(self, exc_name, *args):
        """
        Instantiate an exception
        """
        exc_type = getattr(exceptions, exc_name)
        return exc_type(*args)

#===------------------------------------------------------------------===
# Run
#===------------------------------------------------------------------===

class InterpArgLoader(ArgLoader):

    def load_GlobalValue(self, arg):
        assert not arg.external, "Not supported yet"
        return arg.value.const

    def load_Undef(self, arg):
        return Undef

def run(func, env=None, exc_model=None, _state=None, args=(),
        tracer=tracing.DummyTracer()):
    """
    Interpret function. Raises UncaughtException(exc) for uncaught exceptions
    """
    assert len(func.args) == len(args)

    tracer.push(tracing.Call(func, args))

    # -------------------------------------------------
    # Set up interpreter


    valuemap = dict(zip(func.argnames, args)) # { '%0' : pyval }
    argloader = InterpArgLoader(valuemap)
    interp = Interp(func, env, exc_model or ExceptionModel(),
                    argloader, tracer)
    if env:
        handlers = env.get("interp.handlers") or {}
    else:
        handlers = {}

    # -------------------------------------------------
    # Eval loop

    curblock = None
    while True:
        # -------------------------------------------------
        # Block transitioning

        op = interp.op
        if op.block != curblock:
            interp.blockswitch(curblock, op.block, valuemap)
            curblock = op.block

        # -------------------------------------------------
        # Find handler

        if op.opcode in handlers:
            fn = partial(handlers[op.opcode], interp)
        else:
            fn = getattr(interp, op.opcode)

        # -------------------------------------------------
        # Load arguments

        args = argloader.load_args(op)

        # -------------------------------------------------
        # Execute...

        tracer.push(tracing.Op(op, args))

        oldpc = interp.pc
        try:
            result = fn(*args)
        except UncaughtException, e:
            tracer.push(tracing.Exc(e))
            raise
        valuemap[op.result] = result

        tracer.push(tracing.Res(op, args, result))

        # -------------------------------------------------
        # Advance PC

        if oldpc == interp.pc:
            interp.incr_pc()
        elif interp.pc == -1:
            # Returning...
            tracer.push(tracing.Ret(result))
            return result
