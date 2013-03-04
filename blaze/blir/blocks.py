import opcodes as ops

BASIC  = 0
IFELSE = 1
WHILE  = 2
FOR    = 3

#------------------------------------------------------------------------
# Blocks
#------------------------------------------------------------------------

class Block(object):
    _count = 0

    def __init__(self, label=None):
        self.instrs = []
        self.next_block = None
        self.label = label or self.__class__.__name__
        self.bid = Block._count

        Block._count = Block._count + 1

    @property
    def kind(self):
        return

    def append(self,instr):
        self.instrs.append(instr)

    def __iter__(self):
        return iter(self.instrs)

    def __repr__(self):
        if self.next_block:
            next_id = self.next_block.bid
        else:
            next_id = 'Terminal'

        if self.label:
            return "<block %s id=%d next=%s>" % (self.label, self.bid, next_id)
        else:
            return "<block id=%d next=%>" % (self.bid, next_id)

class BasicBlock(Block):
    kind = BASIC

class IfBlock(Block):
    kind = IFELSE

    def __init__(self):
        super(IfBlock,self).__init__()
        self.true_branch = None
        self.false_branch = None
        self.condition = None

class WhileBlock(Block):
    kind = WHILE

    def __init__(self):
        super(WhileBlock,self).__init__()
        self.body = None
        self.condition = None

class ForBlock(Block):
    kind = FOR

    def __init__(self):
        super(ForBlock,self).__init__()
        self.body = None
        self.var = None
        self.bounds = None
