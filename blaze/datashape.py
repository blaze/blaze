import re

import shlex
import codecs
import string
from collections import deque

from collections import namedtuple

#------------------------------------------------------------------------
# Types
#------------------------------------------------------------------------

Dim = namedtuple('Dim', 'n')
Bit = namedtuple('Bit', 'name, size')

# Variable length string
VString = namedtuple('VString', 'encoding')

# Null-terminated string
CString = namedtuple('CString', 'size, encoding')

# Optional value, for implementation specific notion of optional
Maybe = namedtuple('Maybe', 'ty')

Bool = namedtuple('Bool', ())

Int8  = Bit('int', 8)
Int16 = Bit('int', 16)
Int32 = Bit('int', 32)

UInt8  = Bit('uint', 8)
UInt16 = Bit('uint', 16)
UInt32 = Bit('uint', 32)

Float32 = Bit('float', 32)
Float64 = Bit('float', 64)

String = VString(codecs.lookup('utf8'))

#------------------------------------------------------------------------
# Parser
#------------------------------------------------------------------------

# Rejecting all the machinery in the datashape library, in favor of
# something that I hope is easier for others to wrap their head around
# and build out.

grammar = {
    'int8'    : Int8,
    'int16'   : Int16,
    'int32'   : Int32,

    'unit8'   : UInt8,
    'unit16'  : UInt16,
    'unit32'  : UInt32,

    'float32' : UInt16,
    'float64' : Float64,

    'Maybe'   : Maybe,
}

# Just use a ad-hoc stack machine instead of Ply.

def parse(input):
    lexer = shlex.shlex(input)
    tokens = list(iter(lexer.get_token, ''))

    stack = deque()
    index = 0
    depth = 0

    while 1:
        try:
            token = tokens[index]
            lookahead = tokens[index+1] if (index+1) < len(tokens) else None
        except:
            raise StopIteration


        if token == '(':
            pass

        elif token == ',':
            pass

        elif token == ')':
            ctx = stack.popleft()
            args = []
            while stack:
                args.append(stack.popleft())
            yield ctx(*args)
            depth -= 1

        elif token in grammar and lookahead != '(':
            yield grammar[token]

        elif token in grammar and lookahead == '(':
            stack.append(grammar[token])
            depth += 1

        elif token.isdigit() and depth == 0:
            yield Dim(int(token))

        else:
            stack.append(token)

        index += 1

def dshape(input):
    return tuple(parse(input))

#------------------------------------------------------------------------
# Deconstructors
#------------------------------------------------------------------------

def typeof(dshape):
    return dshape[-1]

def shapeof(dshape):
    return [dim.n for dim in dshape[0:-1]]

if __name__ == '__main__':
    print dshape('3, 3, int32')
