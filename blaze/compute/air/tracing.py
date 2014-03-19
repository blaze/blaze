"""Interpreter tracing of air programs."""

from __future__ import print_function, division, absolute_import
from collections import namedtuple

from .ir import Value
from .utils import nestedmap

#===------------------------------------------------------------------===
# Trace Items
#===------------------------------------------------------------------===

Call = namedtuple('Call', ['func', 'args'])
Op   = namedtuple('Op',   ['op', 'args'])
Res  = namedtuple('Res',  ['op', 'args', 'result'])
Ret  = namedtuple('Ret',  ['result'])
Exc  = namedtuple('Exc',  ['exc'])

#===------------------------------------------------------------------===
# Tracer
#===------------------------------------------------------------------===

def reprobj(obj):
    try: return str(obj)
    except Exception: pass

    try: return repr(obj)
    except Exception: pass

    try: return "Unprintable(%s)" % (vars(obj),)
    except Exception: pass

    return "<unprintable object %s>" % (type(obj),)


def _format_arg(arg):
    if isinstance(arg, Value):
        return repr(arg)
    elif isinstance(arg, dict) and sorted(arg) == ['type', 'value']:
        return '{value=%s}' % (arg['value'],)
    return reprobj(arg)

def _format_args(args):
    return ", ".join(map(str, nestedmap(_format_arg, args)))

class Tracer(object):
    """
    Collects and formats an execution trace when interpreting a program.
    """

    def __init__(self, record=False):
        """
        record: whether to record the trace for later inspection
        """
        self.stmts = []
        self.record = record
        self.beginning = True

        self.callstack = [] # stack of function calls
        self.indent = 0     # indentation level

    @property
    def func(self):
        """Currently executing function"""
        return self.callstack[-1]

    def push(self, item):
        """
        Push a trace item, which is a Stmt or a Call, for processing.
        """
        self.format_item(item)
        if self.record:
            self.stmts.append(item)

    def format_item(self, item):
        """
        Display a single trace item.
        """
        if isinstance(item, Call):
            self.call(item.func)
            self.emit("\n")
            self.emit(" --------> %s(%s)" % (item.func.name,
                                             _format_args(item.args)))
        elif isinstance(item, Op):
            opcode = item.op.opcode
            args = "(%s)" % _format_args(item.args)
            self.emit("%-10s: op %%%-5s: %-80s" % (item.op.block.name,
                                                   item.op.result,
                                                   opcode + args), end='')
        elif isinstance(item, Res):
            if item.result is not None:
                self.emit(" -> %s" % (_format_arg(item.result),))
            else:
                self.emit("")
        elif isinstance(item, Ret):
            self.emit(" <---- (%s) ---- %s" % (_format_arg(item.result),
                                               self.callstack[-1].name))
            self.ret()
        elif isinstance(item, Exc):
            self.emit("\n")
            self.emit(" <-------- propagating %s from %s" % (item.exc,
                                                             self.func.name))
            self.ret()

    def emit(self, s, end="\n"):
        if self.beginning:
            parts = self.func.name.split(".")[-2:]
            name = ".".join(parts)
            print("%-20s: " % name, end="")
        self.beginning = (end == "\n")
        print(" " * self.indent + s, end=end)

    def call(self, func):
        self.callstack.append(func)
        self.indent += 4

    def ret(self):
        self.indent -= 4
        self.callstack.pop()

class DummyTracer(Tracer):

    def format_item(self, item):
        pass

#===------------------------------------------------------------------===
# Utils
#===------------------------------------------------------------------===

def format_stream(stream):
    """
    Format a stream of trace items.
    """
    tracer = Tracer()
    for item in stream:
        tracer.push(item)
