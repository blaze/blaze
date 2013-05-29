from . import btypes
from . import syntax
from . import blocks

from ast import NodeVisitor
from ..py3help import Counter

binary_ops = {
    '+'  : 'BINARY_ADD',
    '-'  : 'BINARY_SUBTRACT',
    '*'  : 'BINARY_MULTIPLY',
    '/'  : 'BINARY_DIVIDE',
    '<<' : 'BINARY_LSHIFT',
    '>>' : 'BINARY_RSHIFT',
}

unary_ops = {
    '+' : 'UNARY_POSITIVE',
    '-' : 'UNARY_NEGATIVE',
    '!' : 'UNARY_NOT'
}

cmp_ops = {
    '<'  : 'lt',
    '<=' : 'le',
    '>'  : 'gt',
    '>=' : 'ge',
    '==' : 'eq',
    '!=' : 'ne',
    '&&' : 'and',
    '||' : 'or'
}

bool_ops = {
    '&&' : 'and',
    '||' : 'or'
}

TEMP_PREFIX = '%'
TEMP_NAMING = "%s%d"

#------------------------------------------------------------------------
# Single Static Assignement
#------------------------------------------------------------------------

class SSARewrite(NodeVisitor):

    def __init__(self):
        self.functions = []
        self.assignments = Counter()
        self.block = blocks.BasicBlock()
        self.start_block = self.block
        self.functions.append(
            ('__module', btypes.void_type, [], self.start_block)
        )

    def atemp(self, val):
        # temporaries are named according to type
        # i%d - integers
        # f%d - floats
        # s%d - string
        # b%d - bool
        name = TEMP_NAMING % (val.name, self.assignments[val.name])
        self.assignments[val.name] += 1
        return name

    def visit_Module(self, node):
        self.visit(node.body)

        self.block.append(('RETURN', btypes.void_type))

    def visit_UnaryOp(self, node):
        self.visit(node.expr)

        target = self.atemp(node.expr.type)
        opcode = unary_ops[node.op]
        inst = (opcode, node.type, node.expr.ssa_name, target)
        self.block.append(inst)

        node.ssa_name = target

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)

        target = self.atemp(node.type)
        opcode = binary_ops[node.op]
        inst = (opcode, node.left.type, node.left.ssa_name, node.right.ssa_name, target)
        self.block.append(inst)

        node.ssa_name = target

    def visit_Tuple(self, node):
        for elt in node.elts:
            self.visit(elt)

    #------------------------------------------------------------------------

    def visit_Const(self,node):
        target = self.atemp(node.type)
        inst = ('LOAD_CONST', node.value, target)
        self.block.append(inst)

        node.ssa_name = target

    def visit_Assign(self,node):
        self.visit(node.expr)
        self.visit(node.store_location)

    def visit_LoadVariable(self, node):
        target = self.atemp(node.type)

        inst = ('LOAD_NAME', node.name, target)
        self.block.append(inst)
        node.ssa_name = target

    def visit_StoreVariable(self, node):
        inst = ('STORE_NAME', node.expr.ssa_name, node.name)
        self.block.append(inst)

    def visit_Project(self, node):
        target = self.atemp(node.type)

        inst = ('PROJECT', node.name, node.proj, target)
        self.block.append(inst)
        node.ssa_name = target

    def visit_LoadIndex(self, node):
        self.visit(node.indexer)
        target = self.atemp(node.type)

        if isinstance(node.indexer, syntax.Tuple):
            inst = ('BINARY_SUBSCR', node.name, node.indexer, target, True)
            self.block.append(inst)
            node.ssa_name = target
        else:
            # indexer is variable or constant
            node.indexvar = node.indexer.ssa_name

            inst = ('BINARY_SUBSCR', node.name, node.indexer.ssa_name, target)
            self.block.append(inst)
            node.ssa_name = target

    def visit_StoreIndex(self, node):
        self.visit(node.indexer)

        # Multiple indices involving coordinate calculations
        # Examples:
        #   A[1,2]
        #   A[i,j]
        if isinstance(node.indexer, syntax.Tuple):
            inst = ('STORE_SUBSCR', node.name, node.indexer, node.expr.ssa_name, True)
            self.block.append(inst)
        # Scalar or single variable index
        # Examples:
        #   A[1]
        #   A[i]
        else:
            node.indexvar = node.indexer.ssa_name
            inst = ('STORE_SUBSCR', node.name, node.indexer.ssa_name, node.expr.ssa_name)
            self.block.append(inst)

    #------------------------------------------------------------------------

    def visit_Print(self, node):
        self.visit(node.expr)

        inst = ('PRINT', node.expr.type, node.expr.ssa_name)
        self.block.append(inst)

    def visit_VarDecl(self, node):
        if node.is_global: # XXX don't use state
            inst = ('GLOBAL', node.type, node.name)
        else:
            inst = ('ALLOC', node.type, node.name)
        self.block.append(inst)
        self.visit(node.expr)

        inst = ('STORE_NAME', node.expr.ssa_name, node.name)
        self.block.append(inst)

    def visit_ConstDecl(self, node):
        self.visit(node.expr)
        assert isinstance(node.expr, syntax.Const)
        val = node.expr.value

        if node.is_global:
            inst = ('GLOBAL', node.expr.type, node.name, val)
        else:
            inst = ('ALLOC', node.expr.type, node.name)

        self.block.append(inst)

    def visit_ExternFuncDecl(self, node):
        self.visit(node.sig)

        args = tuple(arg.type for arg in node.sig.parameters)
        inst = ('DEF_FOREIGN', node.sig.name, node.sig.type, args)
        self.block.append(inst)

    def visit_FunctionCall(self, node):
        args = []
        for arg in node.arglist:
            self.visit(arg)
            args.append(arg.ssa_name)

        target = self.atemp(node.type)
        inst = ('CALL_FUNCTION', node.name, args, target)
        self.block.append(inst)
        node.ssa_name = target

    def visit_Compare(self, node):
        self.visit(node.left)
        self.visit(node.right)

        target = self.atemp(node.type)
        lloc = node.left.ssa_name
        rloc = node.right.ssa_name
        inst = ("COMPARE", node.op, node.left.type, lloc, rloc, target)

        self.block.append(inst)
        node.ssa_name = target

    def visit_IfElseStatement(self, node):
         ifblock = blocks.IfBlock()
         self.block.next_block = ifblock
         self.block = ifblock

         self.visit(node.condition)

         ifblock.testvar = node.condition.ssa_name

         ifblock.true_branch = blocks.BasicBlock()
         self.block = ifblock.true_branch
         self.visit(node.if_statements)

         if node.else_statements:
              ifblock.false_branch = blocks.BasicBlock()
              self.block = ifblock.false_branch
              self.visit(node.else_statements)

         self.block = blocks.BasicBlock()
         ifblock.next_block = self.block

    def visit_Range(self, node):
        self.visit(node.start)
        self.visit(node.stop)

    def visit_ForStatement(self, node):
        forblock = blocks.ForBlock()
        self.block.next_block = forblock
        self.block = forblock

        self.visit(node.iter)

        forblock.start_var = node.iter.start.ssa_name
        forblock.stop_var = node.iter.stop.ssa_name
        forblock.var = node.var

        forblock.body = blocks.BasicBlock()
        self.block = forblock.body
        self.visit(node.body)

        self.block = blocks.BasicBlock()
        forblock.next_block = self.block

    def visit_WhileStatement(self,node):
        whileblock = blocks.WhileBlock()
        self.block.next_block = whileblock
        self.block = whileblock
        self.visit(node.condition)

        whileblock.testvar = node.condition.ssa_name

        whileblock.body = blocks.BasicBlock()
        self.block = whileblock.body
        self.visit(node.statements)

        self.block = blocks.BasicBlock()
        whileblock.next_block = self.block

    def visit_ReturnStatement(self, node):
        self.visit(node.expr)

        inst = ('RETURN', node.expr.type, node.expr.ssa_name)
        self.block.append(inst)

    def visit_FunctionDef(self, node):
        last_block = self.block # pop

        self.block = blocks.BasicBlock()
        retty = node.sig.typename.type
        argtys = [ arg.type for arg in node.sig.parameters]

        fn = (node.sig.name, retty, argtys, self.block)
        self.functions.append(fn)

        for n, parm in enumerate(node.sig.parameters):
            inst = ('LOAD_ARGUMENT', parm.type, parm.name, n)
            self.block.append(inst)

        self.visit(node.statements)

        if node.sig.type == btypes.void_type:
            self.block.append(('RETURN', btypes.void_type))

        self.block = last_block # push

#------------------------------------------------------------------------
# Toplevel
#------------------------------------------------------------------------

def ssa_pass(node):
    gen = SSARewrite()
    gen.visit(node)
    return gen.functions

#------------------------------------------------------------------------
# Block Structure
#------------------------------------------------------------------------

def repr_instr(instr):
    pms = []
    for param in instr:
        if isinstance(param, btypes.Type):
            pms.append(param.name)
        elif isinstance(param, btypes.TParam):
            pms.append('<pty>')
        else:
            pms.append(param)
    return tuple(pms)

class BlockDebug(object):

    def visit(self, block):
        while block is not None:
            name = "visit_%s" % type(block).__name__
            if hasattr(self,name):
                getattr(self,name)(block)
            block = block.next_block

    def visit_BasicBlock(self,block):
        print(str(block.label) + ':')

        for inst in block.instrs:
            if len(inst) == 1:
                print('%-20s' % repr_instr(inst))
            if len(inst) == 2:
                print('\t%-20s %-8s' % repr_instr(inst))
            elif len(inst) == 3:
                print('\t%-20s %-8s %-8s' % repr_instr(inst))
            elif len(inst) == 4:
                print('\t%-20s %-8s %-8s %-8s' % repr_instr(inst))
            elif len(inst) == 5:
                print('\t%-20s %-8s %-8s %-8s %-8s' % repr_instr(inst))

        if block.next_block:
            print('\t%-20s %-8s' % ('JUMP', block.next_block.label))
        else:
            print('\tCBRANCH')

        print("\n")

    def visit_IfBlock(self,block):
        self.visit_BasicBlock(block)
        self.visit(block.true_branch)
        self.visit(block.false_branch)

    def visit_ForBlock(self,block):
        self.visit_BasicBlock(block)
        self.visit(block.body)

    def visit_WhileBlock(self,block):
        self.visit_BasicBlock(block)
        self.visit(block.body)

#------------------------------------------------------------------------
# --ddump-cfg
#------------------------------------------------------------------------

def ddump_blocks(source):
    from . import errors
    from .passes import compile

    with errors.listen():
        ast, env = compile(source)
        functions = env['functions']
        blockprint = BlockDebug()

        if not errors.reported():
            functions = ssa_pass(ast)
            for funcname, retty, argtys, start_block in functions:
                print(str(funcname) + ':')
                blockprint.visit(start_block)
