import sys
import ctypes
from collections import defaultdict

from . import btypes
from . import intrinsics

import llvm.ee as le
import llvm.core as lc
import llvm.passes as lp

from llvm import tbaa
from llvm.core import Module, Builder, Function, Type, Constant

PY3 = bool(sys.version_info[0] == 3)

#------------------------------------------------------------------------
# LLVM Types
#------------------------------------------------------------------------

ptrsize = ctypes.sizeof(ctypes.c_void_p)

int_type   = lc.Type.int()
float_type = lc.Type.double()
bool_type  = lc.Type.int(1)
void_type  = lc.Type.void()
char_type  = lc.Type.int(8)

vec_type = lambda width, elt_type: Type.vector(elt_type, width)

pointer = Type.pointer

any_type = pointer(Type.int(ptrsize))
string_type = pointer(char_type)

# naive array
array_type = lambda elt_type: Type.struct([
    pointer(elt_type), # data      | (<type>)*
    int_type,          # nd        | int
    pointer(int_type), # strides   | int*
], name='ndarray_' + str(elt_type))

intp_type = Type.pointer(int_type)

#------------------------------------------------------------------------
# Array Types
#------------------------------------------------------------------------

# Contiguous or Fortran
# ---------------------
#
# struct {
#    eltype *data;
#    intp shape[nd];
# } contiguous_array(eltype, nd)
#
# struct {
#    eltype *data;
#    diminfo shape[nd];
# } strided_array(eltype, nd)
#
# Structure of Arrays
# -------------------
#
# struct {
#    eltype *data;
#    intp shape[nd];
#    intp stride[nd];
# } strided_soa_array(eltype, nd)
#
# Dimension Info
# -------------
#
# struct {
#   intp dim;
#   intp stride;
#} diminfo

diminfo_type = Type.struct([intp_type,    # shape
                            intp_type     # stride
                            ], name='diminfo')

def ArrayC_Type(eltype):
    return Type.struct([
        pointer(eltype),   # data   | (<type>)*
        intp_type          # shape  | intp
    ], name='Array_C<' + str(eltype) + '>')

def ArrayF_Type(eltype):
    return Type.struct([
        pointer(eltype),    # data   | (<type>)*
        intp_type,          # shape  | intp
    ], name='Array_F<' + str(eltype) + '>')

def ArrayS_Type(eltype):
    return Type.struct([
        pointer(eltype),             # data   | (<type>)*
        Type.array(diminfo_type, 2), # shape  | diminfo
    ], name='Array_S<' + str(eltype) + '>')

poly_arrays = set(['Array_C', 'Array_F', 'Array_S'])

#------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------

false = Constant.int(bool_type, 0)
true  = Constant.int(bool_type, 1)
zero  = Constant.int(int_type, 0)
one   = Constant.int(int_type, 1)
two   = Constant.int(int_type, 2)

#------------------------------------------------------------------------
# Type Relations
#------------------------------------------------------------------------

typemap = {
    'int'    : int_type,
    'float'  : float_type,
    'bool'   : bool_type,
    'str'    : string_type,
    'void'   : void_type,
    'any'    : any_type,
}

ptypemap = {
    'array'   : array_type, # naive array
    'Array_C' : ArrayC_Type,
    'Array_F' : ArrayF_Type,
    'Array_S' : ArrayS_Type,
}

_cache = {}

def arg_typemap(ty):
    if isinstance(ty, btypes.TParam):
        cons = ptypemap[ty.cons.name]
        args = typemap[ty.arg.name]

        if PY3: # hack
           return pointer(cons(args))

        if (cons, args) in _cache:
            ty = _cache[(cons, args)]
        else:
            ty = pointer(cons(args))
            _cache[(cons, args)] = ty
        return ty

    elif isinstance(ty, btypes.Type):
        return typemap[ty.name]
    else:
        raise NotImplementedError

# float instrs
float_instrs = {
    '>'  : lc.FCMP_OGT,
    '<'  : lc.FCMP_OLT,
    '==' : lc.FCMP_OEQ,
    '>=' : lc.FCMP_OGE,
    '<=' : lc.FCMP_OLE,
    '!=' : lc.FCMP_ONE
}

# signed
signed_int_instrs = {
    '==' : lc.ICMP_EQ,
    '!=' : lc.ICMP_NE,
    '<'  : lc.ICMP_SLT,
    '>'  : lc.ICMP_SGT,
    '<=' : lc.ICMP_SLE,
    '>=' : lc.ICMP_SGE,
}

# unsigned
unsigned_int_instrs = {
    '==' : lc.ICMP_EQ,
    '!=' : lc.ICMP_NE,
    '<'  : lc.ICMP_ULT,
    '>'  : lc.ICMP_UGT,
    '<=' : lc.ICMP_ULE,
    '>=' : lc.ICMP_UGE,
}

logic_instrs = set([ '&&', '||' ])

#------------------------------------------------------------------------
# Prelude
#------------------------------------------------------------------------

prelude = {
    'show_int'    : Type.function(void_type, [int_type], False),
    'show_float'  : Type.function(void_type, [float_type], False),
    'show_bool'   : Type.function(void_type, [bool_type], False),
    'show_string' : Type.function(void_type, [string_type], False),
    'show_array'  : Type.function(void_type, [any_type], False),
}

def build_intrinsics(mod):
    # XXX define in seperate module and then link in
    ins = {}

    for name, intr in intrinsics.llvm_intrinsics.items():
         # get the function signature
        name, retty, argtys = getattr(intrinsics, name)

        largtys = list(map(arg_typemap, argtys))
        lretty  = arg_typemap(retty)

        lfunc = Function.intrinsic(mod, intr, largtys)
        ins[name] = lfunc

    return mod, ins

#------------------------------------------------------------------------
# Function Level Codegen
#------------------------------------------------------------------------

CONSTANT_NAMING = '.conststr.%x'
MODULE_NAMING   = '.module.%x'

class LLVMEmitter(object):
    """ LLVM backend for Blir opcodes. """

    def __init__(self, symtab, name="blirkernel"):
        self.module = Module.new(name)
        self.symtab = symtab

        # this is a stack based visitor
        self.block = None
        self.builder = None
        self.function = None
        self.string_count = 0

        self.stack = {} # stack allocated referred by ssa ref
        self.locals = {}
        self.globals = {}
        self.refs = defaultdict(dict) # opaque referencse

        self.intrinsics = {}
        self.add_prelude()

        intmod, intrinsics = build_intrinsics(self.module)
        self.globals.update(intrinsics)

        self.tbaa = tbaa.TBAABuilder.new(self.module, "tbaa.root")

    def visit_op(self, instr):
        for op in instr:
            op_code = op[0]
            op_args = op[1:]
            if hasattr(self, "op_"+op_code):
                getattr(self, "op_"+op_code)(*op_args)
            else:
                raise Exception("Can't translate opcode: op_"+op_code)

    def add_prelude(self):
        for name, function in prelude.items():
            lfunc = Function.new(self.module, function, name)
            lfunc.linkage = lc.LINKAGE_EXTERNAL
            lfunc.visibility = lc.VISIBILITY_HIDDEN
            self.intrinsics[name] = lfunc

    def start_function(self, name, retty, argtys):
        rettype = arg_typemap(retty)
        argtypes = [arg_typemap(arg) for arg in argtys]
        func_type = Type.function(rettype, argtypes, False)

        self.function = Function.new(self.module, func_type, name)
        self.function.add_attribute(lc.ATTR_ALWAYS_INLINE)

        self.block = self.function.append_basic_block("entry")
        self.builder = Builder.new(self.block)
        self.exit_block = self.function.append_basic_block("exit")

        # reset the stack
        self.locals = {}
        self.stack  = {}

        if rettype is not void_type:
            self.locals['retval'] = self.builder.alloca(rettype, "retval")

        self.globals[name] = self.function

    def end_function(self):
        self.builder.position_at_end(self.exit_block)

        if 'retval' in self.locals:
            retval = self.builder.load(self.locals['retval'])
            self.builder.ret(retval)
        else:
            self.builder.ret_void()

    def add_block(self, name):
        return self.function.append_basic_block(name)

    def set_block(self, block):
        self.block = block
        self.builder.position_at_end(block)

    def cbranch(self, cond, true_block, false_block):
        self.builder.cbranch(self.stack[cond], true_block, false_block)

    def branch(self, next_block):
        self.builder.branch(next_block)

    def call(self, fn, argv):
        lfn = self.intrinsics[fn]
        self.builder.call(lfn, argv)

    def const(self, val):
        if isinstance(val, int):
            return Constant.int(int_type, val)
        elif isinstance(val, float):
            return Constant.real(float_type, val)
        elif isinstance(val, bool):
            return Constant.int(bool_type, int(val))
        elif isinstance(val, str):
            return Constant.stringz(val)
        else:
            raise NotImplementedError

    def lookup_var(self, name):
        # XXX replace this with just passing the symbol table
        # from the annotator
        if name in self.locals:
            return self.locals[name]
        else:
            return self.globals[name]

    def puts(self, val):
        if val.type == int_type:
            self.call('print_int', [val])
        elif val.type == float_type:
            self.call('print_float', [val])
        elif val.type == string_type:
            self.call('print_string', [val])
        elif val.type == array_type:
            self.call('print_array', [val])
        else:
            raise NotImplementedError

    #------------------------------------------------------------------------
    # Indexing Coordinates
    #------------------------------------------------------------------------

    def change_coordinates(self, order, arr):
        assert order in ['C', 'F', 'S']

        return ptr

    #------------------------------------------------------------------------
    # Opcodes
    #------------------------------------------------------------------------

    def op_LOAD_CONST(self, value, target):
        if isinstance(value, bool):
            self.stack[target] = Constant.int(bool_type, value)
        elif isinstance(value, int):
            self.stack[target] = Constant.int(int_type, value)
        elif isinstance(value, (float, int)):
            self.stack[target] = Constant.real(float_type, value)
        elif isinstance(value, str):
            content = Constant.stringz(value)
            name = CONSTANT_NAMING % self.string_count
            self.string_count += 1

            globalstr = self.module.add_global_variable(content.type, name)
            globalstr.initializer = content
            globalstr.linkage = lc.LINKAGE_LINKONCE_ODR
            self.stack[target] = globalstr.bitcast(pointer(content.type.element))
        else:
            raise NotImplementedError

    def op_ALLOC(self, ty, name):
        lty = typemap[ty.name]
        var = self.builder.alloca(lty, name)
        self.locals[name] = var

    def op_GLOBAL(self, ty, name, val):
        llstr = self.const(val)
        llty = llstr.type

        var = lc.GlobalVariable.new(self.module, llty, name)
        var.initializer = llstr

        if isinstance(val, str):
            self.globals[name] = var.bitcast(pointer(llstr.type.element))
        else:
            self.globals[name] = var

    def op_PROJECT(self, name, proj_idx, target):
        var = self.lookup_var(name)

        if name in self.refs:
            ref = self.refs[name]['_struct']
            field = self.const(proj_idx)
            ptr = self.builder.gep(ref, [zero, field])

            elt = self.builder.load(ptr)
            self.stack[target] = elt

    def op_LOAD_NAME(self, name, target):
        var = self.lookup_var(name)

        # aggregate type ( reference )
        if name in self.refs:
            ref = self.refs[name]['_struct']
            ptr = self.builder.gep(ref, [zero])
            self.stack[target] = self.builder.bitcast(ptr, any_type)

        # any type ( reference )
        elif var.type == any_type:
            self.stack[target] = var

        # simple type ( value )
        else:
            self.stack[target] = self.builder.load(var, target)

    def op_STORE_NAME(self, source, target):
        self.builder.store(self.stack[source], self.lookup_var(target))

    def op_BINARY_SUBSCR(self, source, index, target, cc=False):
        arr = self.refs[source]

        # assert that the the array contains a field called
        # 'data'
        assert 'data' in arr

        data = arr['data']
        order = arr['_order']

        assert order in ['C', 'F', 'S']

        # Multidimensional indexing
        if cc:
            if order == 'S':
                stride = arr['strides']
                offset = zero

                for i, idx in enumerate(index.elts):
                    ic = self.const(i)
                    ix = self.stack[idx.ssa_name]
                    s = self.builder.load(self.builder.gep(stride, [ic]))

                    tmp = self.buidler.mul(s, ix)
                    loc = self.builder.add(offset, tmp)

            elif order == 'C' or order =='F':
                shape = arr['shape']
                ndim = len(index.elts)
                offset = zero

                for i, idx in enumerate(index.elts):
                    ic = self.const(i)
                    sc = self.builder.load(self.builder.gep(shape, [ic]))
                    if order == 'C' and i == (ndim - 1):
                        tmp = ic
                    elif order == 'F' and i == 0:
                        tmp = ic
                    else:
                        tmp = self.builder.mul(ic, sc)
                    loc = self.builder.add(offset, tmp)

        # Single dimensional indexing
        else:
            loc = self.stack[index]

        val = self.builder.gep(data, [loc])
        elt = self.builder.load(val)
        self.stack[target] = elt

    def op_STORE_SUBSCR(self, source, index, target, cc=False):
        arr = self.refs[source]
        data_ptr = arr['data']

        if cc:
            offset = zero

            for i, idx in enumerate(index.elts):
                ic = self.const(i)
                idxv = self.stack[idx.ssa_name]
                stride = arr['strides']
                stride_elt = self.builder.load(self.builder.gep(stride, [ic]))

                offset = self.builder.add(
                    offset,
                    self.builder.mul(stride_elt, idxv)
                )
        else:
            offset = self.stack[index]

        val = self.builder.gep(data_ptr, [offset])
        self.builder.store(self.stack[target], val)

    def op_BINARY_ADD(self, ty, left, right, val):
        lv = self.stack[left]
        rv = self.stack[right]

        if ty == btypes.int_type:
            self.stack[val] = self.builder.add(lv, rv, val)
        elif ty == btypes.float_type:
            self.stack[val] = self.builder.fadd(lv, rv, val)

    def op_BINARY_SUBTRACT(self, ty, left, right, val):
        lv = self.stack[left]
        rv = self.stack[right]

        if ty == btypes.int_type:
            self.stack[val] = self.builder.sub(lv, rv, val)
        elif ty == btypes.float_type:
            self.stack[val] = self.builder.fsub(lv, rv, val)

    def op_BINARY_MULTIPLY(self, ty, left, right, val):
        lv = self.stack[left]
        rv = self.stack[right]

        if ty == btypes.int_type:
            self.stack[val] = self.builder.mul(lv, rv, val)
        elif ty == btypes.float_type:
            self.stack[val] = self.builder.fmul(lv, rv, val)

    def op_BINARY_DIVIDE(self, ty, left, right, val):
        lv = self.stack[left]
        rv = self.stack[right]

        if ty == btypes.int_type:
            self.stack[val] = self.builder.sdiv(lv, rv, val)
        elif ty == btypes.float_type:
            self.stack[val] = self.builder.fdiv(lv, rv, val)

    def op_BINARY_LSHIFT(self, ty, left, right, val):
        lv = self.stack[left]
        rv = self.stack[right]

        if ty == btypes.int_type:
            self.stack[val] = self.builder.shl(lv, rv, val)

    def op_BINARY_RSHIFT(self, ty, left, right, val):
        lv = self.stack[left]
        rv = self.stack[right]

        if ty == btypes.int_type:
            self.stack[val] = self.builder.ashr(lv, rv, val)

    #------------------------------------------------------------------------
    # Unary Operators
    #------------------------------------------------------------------------

    def op_UNARY_POSITIVE(self, ty, source, target):
        self.stack[target] = self.stack[source]

    def op_UNARY_NEGATIVE(self, ty, source, target):

        if ty == btypes.int_type:
            self.stack[target] = self.builder.sub(
                Constant.int(int_type, 0),
                self.stack[source],
                target
            )
        elif ty == btypes.float_type:
            self.stack[target] = self.builder.fsub(
                Constant.real(float_type, 0.0),
                self.stack[source],
                target
            )

    def op_UNARY_NOT(self, ty, source, val):
        if ty == btypes.int_type:
            self.stack[val] = self.builder.icmp(lc.ICMP_EQ, self.stack[source], zero, val)
        elif ty == btypes.bool_type:
            self.stack[val] = self.builder.not_(self.stack[source])

    def op_COMPARE(self, op, ty, left, right, val):
        lv = self.stack[left]
        rv = self.stack[right]

        if ty == btypes.int_type:
            instr = signed_int_instrs[op]
            self.stack[val] = self.builder.icmp(instr, lv, rv, val)

        elif ty == btypes.float_type:
            instr = float_instrs[op]
            self.stack[val] = self.builder.fcmp(instr, lv, rv, val)

        if ty == btypes.bool_type:
            if op in logic_instrs:
                if op == '&&':
                    self.stack[val] = self.builder.and_(lv, rv, val)
                elif op == '||':
                    self.stack[val] = self.builder.or_(lv, rv, val)
            else:
                instr = signed_int_instrs[op]
                self.stack[val] = self.builder.icmp(instr, lv, rv, val)

    #------------------------------------------------------------------------
    # Ref Functions
    #------------------------------------------------------------------------

    def op_REF(self, ty, source, target):
        self.stack[target] = self.builder.gep([zero], source)

    #------------------------------------------------------------------------
    # Vec Functions
    #------------------------------------------------------------------------

    def op_VEC(self, ty, width, source, target):
        self.stack[target] = self.builder.bitcast(source,
                vec_type(width, ty))

    #------------------------------------------------------------------------
    # Show Functions
    #------------------------------------------------------------------------

    def op_PRINT(self, ty, source):
        if ty == btypes.int_type:
            self.call('show_int', [self.stack[source]])
        elif ty == btypes.float_type:
            self.call('show_float', [self.stack[source]])
        elif ty == btypes.string_type:
            self.call('show_string', [self.stack[source]])
        elif ty == btypes.bool_type:
            tmp = self.builder.zext(self.stack[source], bool_type)
            self.call('show_bool', [tmp])
        elif isinstance(ty, btypes.TParam) and ty.cons.type == btypes.array_type:
            self.call('show_array', [self.stack[source]])
        else:
            raise NotImplementedError

    #------------------------------------------------------------------------
    # Extern Functions
    #------------------------------------------------------------------------

    def op_DEF_FOREIGN(self, name, retty, argtys):
        largtys = list(map(arg_typemap, argtys))
        lretty  = arg_typemap(retty)

        func_type = Type.function(lretty, largtys, False)
        self.globals[name] = Function.new(self.module, func_type, name)

    def op_CALL_FUNCTION(self, funcname, args, target):
        func = self.globals[funcname]
        argvals = [self.stack[name] for name in args]
        self.stack[target] = self.builder.call(func, argvals)

    #------------------------------------------------------------------------
    # Return
    #------------------------------------------------------------------------

    def op_RETURN(self, ty, val=None):
        if ty == btypes.void_type:
            self.builder.branch(self.exit_block)
        else:
            self.builder.store(self.stack[val], self.locals['retval'])
            self.builder.branch(self.exit_block)

    #------------------------------------------------------------------------
    # Argument Variables
    #------------------------------------------------------------------------

    def op_LOAD_ARGUMENT(self, ty, name, num):
        # aggregate types, pass by reference
        arg = self.function.args[num]

        if isinstance(ty, btypes.TParam):

            # primitive array
            # -------------
            if ty.cons.name == 'array':
                struct_ptr = arg

                data    = self.builder.gep(struct_ptr, [zero, zero], name=(name + '_data'))
                dims    = self.builder.gep(struct_ptr, [zero, one], name=(name + '_dims'))
                strides = self.builder.gep(struct_ptr, [zero, two], name=(name + '_strides'))

                self.refs[name]['_order'] = 'C' # hack
                self.refs[name]['_struct'] = struct_ptr
                self.refs[name]['_dtype']  = typemap[ty.arg.type.name]
                self.refs[name]['data']    = self.builder.load(data)
                self.refs[name]['dims']    = self.builder.load(dims)
                self.refs[name]['strides'] = self.builder.load(strides)

                self.locals[name] = self.refs[name]

            elif ty.cons.name in poly_arrays:
                struct_ptr = arg
                self.refs[name]['_struct'] = struct_ptr
                self.refs[name]['_order']  = ty.cons.type.order
                self.refs[name]['_dtype']  = typemap[ty.arg.type.name]

                fields = ty.cons.type.fields

                for fname, (idx, field) in fields.iteritems():
                    proj_idx = self.const(idx)
                    field = self.builder.gep(struct_ptr, [zero, proj_idx],
                            name=fname)
                    self.refs[name][fname] = self.builder.load(field)

                self.locals[name] = self.refs[name]

            else:
                raise NotImplementedError

        # opaque any types
        elif arg.type == any_type:
            self.locals[name] = arg

        # concrete types, pass by value
        else:
            var = self.builder.alloca(arg.type, name)
            self.builder.store(arg, var)
            self.locals[name] = var

#------------------------------------------------------------------------
# Module Level Codegen
#------------------------------------------------------------------------

class BlockEmitter(object):

    def __init__(self, generator):
        self.cgen = generator

    def visit(self, block):
        while block is not None:
            name = "visit_%s" % type(block).__name__
            if hasattr(self, name):
                getattr(self, name)(block)
            block = block.next_block

    def generate_function(self, name, retty, argtyp, start_block):
        self.cgen.start_function(name, retty, argtyp)
        self.visit(start_block)
        self.cgen.end_function()
        return self.cgen.function

    def visit_BasicBlock(self, block):
        self.cgen.visit_op(block)

    def visit_IfBlock(self, block):
        self.cgen.visit_op(block)

        true_block  = self.cgen.add_block("if.then")
        false_block = self.cgen.add_block("if.else")
        endif_block = self.cgen.add_block("if.end")

        self.cgen.cbranch(block.testvar, true_block, false_block)

        # Visit the true-branch
        self.cgen.set_block(true_block)
        self.visit(block.true_branch)
        self.cgen.branch(endif_block)

        # Visit the false-branch
        self.cgen.set_block(false_block)
        self.visit(block.false_branch)
        self.cgen.branch(endif_block)

        self.cgen.set_block(endif_block)

    def visit_ForBlock(self, block):
        self.cgen.visit_op(block)

        # convienance aliases for the cgen parent
        builder = self.cgen.builder
        const   = self.cgen.const

        start = block.start_var
        stop  = block.stop_var
        step  = 1

        init_block = self.cgen.add_block('for.init')
        test_block = self.cgen.add_block('for.cond')
        body_block = self.cgen.add_block('for.body')
        end_block  = self.cgen.add_block("for.end")

        self.cgen.branch(init_block)
        self.cgen.set_block(init_block)

        # ------------------------------------------
        varname = block.var
        inc = builder.alloca(int_type, varname)
        self.cgen.locals[varname] = inc
        builder.store(self.cgen.stack[start], inc)
        # ------------------------------------------
        self.cgen.branch(test_block)
        self.cgen.set_block(test_block)
        # ------------------------------------------

        stopv = self.cgen.stack[stop]
        cond = builder.icmp(lc.ICMP_SLT, builder.load(inc), stopv)
        builder.cbranch(cond, body_block, end_block)

        # Generate the loop body
        self.cgen.set_block(body_block)
        self.visit(block.body)
        succ = builder.add(const(step), builder.load(inc))
        builder.store(succ, inc)

        self.cgen.branch(test_block)
        self.cgen.set_block(end_block)

    def visit_WhileBlock(self, block):
        test_block = self.cgen.add_block("while.cond")

        # ------------------------------------------
        self.cgen.branch(test_block)
        self.cgen.set_block(test_block)
        # ------------------------------------------

        self.cgen.visit_op(block)

        loop_block = self.cgen.add_block("while.body")
        after_loop = self.cgen.add_block("while.end")

        self.cgen.cbranch(block.testvar, loop_block, after_loop)

        # Generate the loop body
        self.cgen.set_block(loop_block)
        self.visit(block.body)
        self.cgen.branch(test_block)

        self.cgen.set_block(after_loop)
