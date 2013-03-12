opmap = {}
opname = {}

def def_op(name, op):
    opname[op] = name
    opmap[name] = op

#------------------------------------------------------------------------
# Instructions
#------------------------------------------------------------------------

# ALLOC
# ARRAYLOAD
# ARRAYSTORE
# BINARY_ADD
# BINARY_DIVIDE
# BINARY_MULTIPLY
# BINARY_SUBTRACT
# CALL_FUNCTION
# COMPARE
# DEF_FOREIGN
# GLOBAL
# LOAD
# LOAD_ARGUMENT
# LOAD_CONST
# PRINT
# RETURN
# STORE
# UNARY_NEGATIVE
# UNARY_NOT
# UNARY_POSITIVE

class Instruction(object):

    def __init__(self, opcode, arg=None):
        self.opcode = opcode
        self.arg = arg
        self.have_argument = arg is not None

    @property
    def opname(self):
        return opname[self.opcode]

    def __iter__(self):
        if self.arg is not 0:
            return (self.opcode, self.arg)
        else:
            return (self.opcode, None)

    def __repr__(self):
        data = [opname[self.opcode]]
        template = "<%s"

        if self.have_argument:
            data.append(self.arg)
            template += " %i"

        template += ">"
        return template % tuple(data)
